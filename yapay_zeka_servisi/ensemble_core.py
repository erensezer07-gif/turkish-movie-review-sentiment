"""BERT + TF-IDF ensemble: model yükleme, inference, karar mantığı.

Streamlit'e bağımlı değil; Django ve standalone scriptlerden import edilebilir.
TF-IDF eksikse BERT-only fallback'e düşer.
"""
import os
import torch
import numpy as np
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from .nlp_utils import temizle_tek, temizle_liste
except ImportError:
    from nlp_utils import temizle_tek, temizle_liste

from . import rules
from .rules import (
    validate_text, check_guardrails, is_neutral_like, has_soft_neutral_signal,
    split_on_sarcasm, has_sarcasm_negative_tail,
    BERT_NEUTRAL_LOW, BERT_NEUTRAL_HIGH,
)

# ---------------------------------------------------------------------
# 1) PATH'LER
# ---------------------------------------------------------------------
SERVICE_DIR = Path(__file__).resolve().parent
TFIDF_BUNDLE_PATH = SERVICE_DIR / "film_tfidf_3cls.pkl"

_bert_model_path_env = os.getenv("BERT_MODEL_PATH")
BERT_MODEL_PATH = (
    Path(_bert_model_path_env) if _bert_model_path_env else (SERVICE_DIR / "benim_bert_modelim_3cls_v2")
)

print("[OK] SERVICE_DIR:", SERVICE_DIR)
print("[OK] BERT_MODEL_PATH:", BERT_MODEL_PATH, "exists=", BERT_MODEL_PATH.exists())
print("[OK] TFIDF_PATH:", TFIDF_BUNDLE_PATH, "exists=", TFIDF_BUNDLE_PATH.exists())

# 3-sınıf etiketler (BERT load sonrası override)
LABELS = ["OLUMSUZ", "NÖTR", "OLUMLU"]
ID2LABEL = {0: "OLUMSUZ", 1: "NÖTR", 2: "OLUMLU"}
LABEL2ID = {"OLUMSUZ": 0, "NÖTR": 1, "OLUMLU": 2}

# ---------------------------------------------------------------------
# 2) STATS HOOK (UI'lar tarafından enjekte edilebilir; default no-op)
# ---------------------------------------------------------------------
_stats_hook = None


def set_stats_hook(fn):
    """UI'lar burada bir callback set edebilir: fn(key: str, n: int=1)."""
    global _stats_hook
    _stats_hook = fn


def _bump(key: str, n: int = 1):
    if _stats_hook is not None:
        try:
            _stats_hook(key, n)
        except Exception:
            pass


# ---------------------------------------------------------------------
# 3) LOADERS
# ---------------------------------------------------------------------
def load_tfidf_bundle(path: Path):
    if not path.exists():
        return None, f"TF-IDF model bulunamadı: {path}"

    def _normalize_obj(obj):
        if isinstance(obj, dict):
            if "model" not in obj:
                for k in ["pipeline", "clf", "estimator"]:
                    if k in obj:
                        obj["model"] = obj[k]
                        break
            if "model" not in obj:
                return None, "TF-IDF bundle dict ama 'model' anahtarı yok."
            return obj, None
        if hasattr(obj, "predict_proba"):
            return {"model": obj}, None
        return None, "TF-IDF dosyası tanınamadı."

    try:
        return _normalize_obj(joblib.load(path))
    except AttributeError:
        # Pickle eski referansları arıyor olabilir; nlp_utils'i __main__ üzerinden eriştir.
        import __main__
        __main__.temizle_liste = temizle_liste
        __main__.temizle_tek = temizle_tek
        try:
            return _normalize_obj(joblib.load(path))
        except Exception as e:
            return None, f"TF-IDF bundle yüklenemedi: {e}"
    except Exception as e:
        return None, f"TF-IDF bundle yüklenemedi: {e}"


def detect_label_mapping_3cls(model):
    id2label = getattr(model.config, "id2label", {}) or {}
    found = {"neg": None, "neutral": None, "pos": None}
    for k, v in id2label.items():
        kk = int(k) if str(k).isdigit() else k
        name = (
            str(v).strip().lower()
            .replace("ı", "i").replace("ö", "o").replace("ü", "u")
            .replace("ş", "s").replace("ç", "c").replace("ğ", "g")
        )
        if "olumsuz" in name or "negative" in name or name in {"neg", "label_0"}:
            found["neg"] = int(kk)
        elif "notr" in name or "neutral" in name or name in {"neu", "label_1"}:
            found["neutral"] = int(kk)
        elif "olumlu" in name or "positive" in name or name in {"pos", "label_2"}:
            found["pos"] = int(kk)
    assumed = False
    if any(v is None for v in found.values()):
        found = {"neg": 0, "neutral": 1, "pos": 2}
        assumed = True
    return found, assumed


def load_bert(model_path: Path):
    if not model_path.exists():
        return None, None, None, f"BERT klasörü bulunamadı: {model_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
        model.to(device)
        model.eval()

        # Dynamic Quantization (RAM Optimizasyonu)
        try:
            print("[INFO] Applying Dynamic Quantization (int8)...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("[OK] Quantization applied successfully.")
        except Exception as e:
            print(f"[WARN] Quantization failed: {e}")

        print("[OK] BERT LOADED num_labels:", model.config.num_labels)
        print("[OK] BERT LOADED id2label:", model.config.id2label)

        global ID2LABEL, LABEL2ID, LABELS
        if model.config.id2label:
            ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}
            LABEL2ID = {v: int(k) for k, v in model.config.id2label.items()}
            LABELS = [ID2LABEL.get(i, f"LABEL_{i}") for i in range(model.config.num_labels)]
            print("[OK] Label map config'ten alındı:", ID2LABEL)

        mapping, assumed = detect_label_mapping_3cls(model)
        meta = {
            "device": device,
            "idx_neg": mapping["neg"],
            "idx_neu": mapping["neutral"],
            "idx_pos": mapping["pos"],
            "assumed_mapping": assumed,
        }
        return tokenizer, model, meta, None
    except Exception as e:
        return None, None, None, f"BERT yüklenemedi: {e}"


# ---------------------------------------------------------------------
# 4) MODÜL SEVİYESİNDE TEK SEFERLİK YÜKLEME
# ---------------------------------------------------------------------
tfidf_bundle, tfidf_err = load_tfidf_bundle(TFIDF_BUNDLE_PATH)
tokenizer, bert_model, bert_meta, bert_err = load_bert(BERT_MODEL_PATH)

# Self-test
if tokenizer is not None and bert_model is not None:
    _test_text = "çok ama çok iyi"
    _test_inp = tokenizer(_test_text, return_tensors="pt", truncation=True, max_length=256)
    _test_inp = {k: v.to(bert_meta["device"]) for k, v in _test_inp.items()}
    with torch.no_grad():
        _test_out = bert_model(**_test_inp)
    _test_pred = int(torch.argmax(_test_out.logits, dim=-1).item())
    print(f"[TEST] SELFTEST: '{_test_text}' pred={_test_pred} label={ID2LABEL.get(_test_pred, _test_pred)}")
else:
    print("[WARN] BERT yüklenemedi, self-test atlandı.")


# ---------------------------------------------------------------------
# 5) BERT INFERENCE (Chunking)
# ---------------------------------------------------------------------
def _split_into_token_chunks(tokenizer, text: str, max_length: int, stride: int):
    enc = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = enc["input_ids"]
    if len(ids) <= max_length - 2:
        return [text]

    step = max(1, (max_length - 2) - stride)
    chunks = []
    for start in range(0, len(ids), step):
        end = min(len(ids), start + (max_length - 2))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end >= len(ids):
            break
    return chunks


def _aggregate_chunk_probs(probs_chunks: np.ndarray, mode: str = "mean_max"):
    if probs_chunks.shape[0] == 1:
        p = probs_chunks[0]
        return p / (p.sum() + 1e-12)
    mean_p = probs_chunks.mean(axis=0)
    max_p = probs_chunks.max(axis=0)
    if mode == "mean":
        p = mean_p
    elif mode == "max":
        p = max_p
    else:
        p = 0.70 * mean_p + 0.30 * max_p
    return p / (p.sum() + 1e-12)


def bert_predict_proba_batch(
    texts, tokenizer, model, meta,
    batch_size=16, max_length=192, chunk_stride=64, chunk_mode="mean_max",
):
    device = meta["device"]
    probs_out = np.zeros((len(texts), 3), dtype=float)

    all_chunks = []
    owners = []
    for i, t in enumerate(texts):
        t = "" if t is None else str(t)
        chunks = _split_into_token_chunks(tokenizer, t, max_length=int(max_length), stride=int(chunk_stride))
        for ch in chunks:
            all_chunks.append(ch)
            owners.append(i)

    owners = np.array(owners, dtype=int)
    probs_chunks_all = np.zeros((len(all_chunks), 3), dtype=float)

    for start in range(0, len(all_chunks), batch_size):
        end = min(len(all_chunks), start + batch_size)
        batch_texts = all_chunks[start:end]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            padding=True, max_length=int(max_length),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        probs_chunks_all[start:end, 0] = probs[:, meta["idx_neg"]]
        probs_chunks_all[start:end, 1] = probs[:, meta["idx_neu"]]
        probs_chunks_all[start:end, 2] = probs[:, meta["idx_pos"]]

    for i in range(len(texts)):
        idxs = np.where(owners == i)[0]
        probs_out[i] = _aggregate_chunk_probs(probs_chunks_all[idxs], mode=str(chunk_mode))

    return probs_out


# ---------------------------------------------------------------------
# 6) TF-IDF INFERENCE (None-safe)
# ---------------------------------------------------------------------
def _reorder_probs_to_012(probs: np.ndarray, classes_) -> np.ndarray:
    cls = list(classes_) if classes_ is not None else None
    if cls == [0, 1, 2]:
        return probs
    mapping = {}
    if cls is not None:
        for idx, lab in enumerate(cls):
            try:
                lab_int = int(lab)
            except Exception:
                lab_int = None
            if lab_int in (0, 1, 2):
                mapping[lab_int] = idx
    out = np.zeros((probs.shape[0], 3), dtype=float)
    for target in [0, 1, 2]:
        src_idx = mapping.get(target, target if target < probs.shape[1] else 0)
        out[:, target] = probs[:, src_idx]
    return out


def tfidf_predict_proba(texts):
    """TF-IDF tahmin. Bundle yoksa None döner; caller BERT-only fallback yapar."""
    if tfidf_bundle is None or "model" not in tfidf_bundle:
        return None
    model = tfidf_bundle["model"]
    probs = model.predict_proba(texts)
    if probs.shape[1] == 2:
        out = np.zeros((len(texts), 3), dtype=float)
        out[:, 0] = probs[:, 0]
        out[:, 2] = probs[:, 1]
        return out
    return _reorder_probs_to_012(probs, getattr(model, "classes_", None))


# ---------------------------------------------------------------------
# 7) ENSEMBLE
# ---------------------------------------------------------------------
def pick_label_from_probs(p3: np.ndarray):
    idx = int(np.argmax(p3))
    return ID2LABEL[idx], idx, float(p3[idx])


def apply_neutral_band(p3: np.ndarray, low: float, high: float):
    return bool(low <= float(p3[1]) <= high)


def top2_info(p3: np.ndarray):
    order = np.argsort(p3)[::-1]
    top1, top2 = float(p3[order[0]]), float(p3[order[1]])
    return int(order[0]), int(order[1]), top1, top2, top1 - top2


def _mix_probs(p_tfidf, p_bert, tw, bw):
    """TF-IDF None ise BERT-only; aksi halde ağırlıklı karışım."""
    if p_tfidf is None:
        return np.array(p_bert, dtype=float)
    if tw + bw <= 0:
        tw, bw = 0.5, 0.5
    p = tw * p_tfidf + bw * p_bert
    return p / (p.sum() + 1e-12)


def ensemble_single(
    text,
    use_guardrail=True,
    guard_cutoff=0.85,
    neutral_on=True,
    use_neutral_band=False,
    bert_neutral_low=BERT_NEUTRAL_LOW,
    bert_neutral_high=BERT_NEUTRAL_HIGH,
    tfidf_weight=0.30,
    bert_weight=0.70,
    bert_batch_size=16,
    bert_max_len=192,
    debug_mode=False,
    uncertain_to_neutral_on=False,
    conf_threshold=0.50,
    margin_threshold=0.15,
    min_neutral_prob=0.25,
):
    logs = []
    try:
        _bump("total")
        validated_text, validation_msg = validate_text(text)
        if validated_text is None:
            _bump("error")
            return "GEÇERSİZ", 0.0, "Error", {"error": validation_msg}

        # SARCASM
        s_head, s_tail, s_marker = split_on_sarcasm(validated_text)
        if s_tail and len(s_tail) >= 5:
            if debug_mode:
                logs.append(f"Sarcasm marker: '{s_marker}' tail: '{s_tail[:60]}'")
            if has_sarcasm_negative_tail(s_tail):
                _bump("guardrail")
                return "OLUMSUZ", 0.99, "SarcasmRule", {
                    "marker": s_marker, "reason": "Negative Tail Keywords",
                    "logs": logs if debug_mode else None,
                }
            if tokenizer is not None and bert_model is not None:
                try:
                    p_tail = bert_predict_proba_batch(
                        [s_tail], tokenizer, bert_model, bert_meta,
                        batch_size=1, max_length=int(bert_max_len),
                    )[0]
                    tail_label, _, tail_conf = pick_label_from_probs(p_tail)
                    if tail_label == "OLUMSUZ" and tail_conf >= 0.60:
                        _bump("guardrail")
                        return "OLUMSUZ", float(tail_conf), "Sarcasm->BERTTail", {
                            "marker": s_marker, "tail_text": s_tail[:50],
                            "logs": logs if debug_mode else None,
                        }
                except Exception as e:
                    if debug_mode:
                        logs.append(f"Sarcasm BERT check failed: {e}")

        # GUARDRAIL
        if use_guardrail:
            hint = check_guardrails(validated_text, cutoff=guard_cutoff)
            if debug_mode:
                logs.append(f"Guardrail hint: {hint}")
            if hint == "neutral":
                _bump("guardrail")
                return "NÖTR", 0.99, "Guardrail", {"hint": "neutral", "logs": logs if debug_mode else None}
            if hint == "neg":
                _bump("guardrail")
                return "OLUMSUZ", 0.99, "Guardrail", {"hint": "neg", "logs": logs if debug_mode else None}
            if hint == "pos":
                _bump("guardrail")
                return "OLUMLU", 0.99, "Guardrail", {"hint": "pos", "logs": logs if debug_mode else None}

        # NEUTRAL RULE
        if neutral_on:
            is_neu, reason = is_neutral_like(validated_text, return_reason=True)
            if debug_mode:
                logs.append(f"NeutralRule: {is_neu} ({reason})")
            if is_neu:
                _bump("neutralrule")
                return "NÖTR", 0.99, "NeutralRule", {"reason": reason, "logs": logs if debug_mode else None}

        # MODEL TAHMİNLERİ
        if tokenizer is None or bert_model is None:
            raise RuntimeError("BERT mevcut değil; ensemble çalıştırılamaz.")

        p_tfidf_raw = tfidf_predict_proba([validated_text])
        p_tfidf = p_tfidf_raw[0] if p_tfidf_raw is not None else None
        if p_tfidf is not None:
            _bump("tfidf")
        if debug_mode:
            if p_tfidf is None:
                logs.append("TFIDF: yok (BERT-only fallback)")
            else:
                logs.append(f"TFIDF: N={p_tfidf[0]:.2f} U={p_tfidf[1]:.2f} P={p_tfidf[2]:.2f}")

        p_bert = bert_predict_proba_batch(
            [validated_text], tokenizer, bert_model, bert_meta,
            batch_size=1, max_length=int(bert_max_len),
        )[0]
        _bump("bert")
        if debug_mode:
            logs.append(f"BERT : N={p_bert[0]:.2f} U={p_bert[1]:.2f} P={p_bert[2]:.2f}")

        if use_neutral_band and apply_neutral_band(p_bert, bert_neutral_low, bert_neutral_high):
            _bump("ensemble")
            return "NÖTR", float(p_bert[1]), "Neutral-BERTBand", {
                "p_bert": p_bert.tolist(), "logs": logs if debug_mode else None,
            }

        p_mix = _mix_probs(p_tfidf, p_bert, float(tfidf_weight), float(bert_weight))
        if debug_mode:
            logs.append(f"MIX  : N={p_mix[0]:.2f} U={p_mix[1]:.2f} P={p_mix[2]:.2f}")

        if uncertain_to_neutral_on:
            _, _, top1, top2, margin = top2_info(p_mix)
            p_neu = float(p_mix[1])
            soft_neu = has_soft_neutral_signal(validated_text)

            conf_eff = max(float(conf_threshold), 0.57) if soft_neu else float(conf_threshold)
            margin_eff = max(float(margin_threshold), 0.22) if soft_neu else float(margin_threshold)
            min_neu_eff = min(float(min_neutral_prob), 0.18) if soft_neu else float(min_neutral_prob)

            if (top1 < conf_eff) and (margin < margin_eff) and (p_neu >= min_neu_eff):
                _bump("uncertainneutral")
                return "NÖTR", p_neu, "Uncertain→Neutral", {"logs": logs if debug_mode else None}

        label, _, conf = pick_label_from_probs(p_mix)
        _bump("ensemble")
        kaynak = "Ensemble" if p_tfidf is not None else "BERT-only"
        return label, conf, kaynak, {"p_mix": p_mix.tolist(), "logs": logs if debug_mode else None}

    except Exception as e:
        _bump("error")
        return "HATA", 0.0, "Error", {"error": str(e), "logs": logs if debug_mode else None}


def ensemble_batch(
    texts,
    use_guardrail=True,
    guard_cutoff=0.85,
    neutral_on=True,
    use_neutral_band=False,
    bert_neutral_low=BERT_NEUTRAL_LOW,
    bert_neutral_high=BERT_NEUTRAL_HIGH,
    tfidf_weight=0.30,
    bert_weight=0.70,
    bert_batch_size=16,
    bert_max_len=192,
    progress_callback=None,
    uncertain_to_neutral_on=True,
    conf_threshold=0.50,
    margin_threshold=0.15,
    min_neutral_prob=0.25,
):
    try:
        n = len(texts)
        labels = np.array([""] * n, dtype=object)
        conf_scores = np.zeros(n, dtype=float)
        sources = np.array([""] * n, dtype=object)
        unresolved = np.ones(n, dtype=bool)
        processed = [None] * n

        for i in range(n):
            _bump("total")
            vt, _ = validate_text(texts[i])
            if vt is None:
                labels[i] = "GEÇERSİZ"
                conf_scores[i] = 0.0
                sources[i] = "Error"
                unresolved[i] = False
                _bump("error")
            else:
                processed[i] = vt

        if progress_callback:
            progress_callback(0.05)

        if use_guardrail:
            for i in range(n):
                if not unresolved[i]:
                    continue
                h = check_guardrails(processed[i], cutoff=guard_cutoff)
                if h in ["neg", "pos", "neutral"]:
                    mapping = {"neg": "OLUMSUZ", "pos": "OLUMLU", "neutral": "NÖTR"}
                    labels[i] = mapping[h]
                    conf_scores[i] = 0.99
                    sources[i] = "Guardrail"
                    unresolved[i] = False
                    _bump("guardrail")

        if progress_callback:
            progress_callback(0.20)

        if neutral_on:
            for i in range(n):
                if not unresolved[i]:
                    continue
                if is_neutral_like(processed[i]):
                    labels[i] = "NÖTR"
                    conf_scores[i] = 0.99
                    sources[i] = "NeutralRule"
                    unresolved[i] = False
                    _bump("neutralrule")

        if progress_callback:
            progress_callback(0.35)

        idxs = np.where(unresolved)[0]
        if len(idxs) > 0:
            if tokenizer is None or bert_model is None:
                raise RuntimeError("BERT mevcut değil; ensemble çalıştırılamaz.")

            sel_texts = [processed[i] for i in idxs]
            p_tfidf_all = tfidf_predict_proba(sel_texts)
            if p_tfidf_all is not None:
                _bump("tfidf", len(idxs))

            p_bert_all = bert_predict_proba_batch(
                sel_texts, tokenizer, bert_model, bert_meta,
                batch_size=max(1, int(bert_batch_size)),
                max_length=int(bert_max_len),
            )
            _bump("bert", len(idxs))

            tw, bw = float(tfidf_weight), float(bert_weight)

            for k, i in enumerate(idxs):
                p_tfidf = p_tfidf_all[k] if p_tfidf_all is not None else None
                p_bert = p_bert_all[k]

                if use_neutral_band and apply_neutral_band(p_bert, bert_neutral_low, bert_neutral_high):
                    labels[i] = "NÖTR"
                    conf_scores[i] = float(p_bert[1])
                    sources[i] = "Neutral-BERTBand"
                    unresolved[i] = False
                    _bump("ensemble")
                    continue

                p_mix = _mix_probs(p_tfidf, p_bert, tw, bw)

                if uncertain_to_neutral_on:
                    _, _, t1, _, marg = top2_info(p_mix)
                    p_neu = float(p_mix[1])
                    soft_neu = has_soft_neutral_signal(processed[i])

                    ce = max(float(conf_threshold), 0.57) if soft_neu else float(conf_threshold)
                    me = max(float(margin_threshold), 0.22) if soft_neu else float(margin_threshold)
                    mn = min(float(min_neutral_prob), 0.18) if soft_neu else float(min_neutral_prob)

                    if (t1 < ce) and (marg < me) and (p_neu >= mn):
                        labels[i] = "NÖTR"
                        conf_scores[i] = p_neu
                        sources[i] = "Uncertain→Neutral"
                        unresolved[i] = False
                        _bump("uncertainneutral")
                        continue

                lab, _, conf = pick_label_from_probs(p_mix)
                labels[i] = lab
                conf_scores[i] = conf
                sources[i] = "Ensemble" if p_tfidf is not None else "BERT-only"
                unresolved[i] = False
                _bump("ensemble")

        if progress_callback:
            progress_callback(1.0)

        return labels.tolist(), conf_scores, sources.tolist()

    except Exception as e:
        return ["HATA"] * len(texts), np.zeros(len(texts)), [f"Error: {e}"] * len(texts)

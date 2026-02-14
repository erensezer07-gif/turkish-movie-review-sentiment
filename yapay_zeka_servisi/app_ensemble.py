import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import difflib
import joblib
import re
try:
    from .nlp_utils import temizle_tek, temizle_liste
except ImportError:
    from nlp_utils import temizle_tek, temizle_liste

import sys

RUNNING_IN_STREAMLIT = any("streamlit" in arg.lower() for arg in sys.argv)

if RUNNING_IN_STREAMLIT:
    import streamlit as st
else:
    # Uvicorn/FastAPI import ederken Streamlit hiç devreye girmesin
    class _DummyStreamlit:
        def cache_resource(self, func=None, **kwargs):
            if func is None:
                def deco(f):
                    return f
                return deco
            return func

        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop

    st = _DummyStreamlit()

# Plotly (grafik) - yoksa fallback
try:
    import plotly.express as px
except Exception:
    px = None

# ---------------------------------------------------------------------
# 1) SAYFA AYARLARI
# ---------------------------------------------------------------------
if RUNNING_IN_STREAMLIT:
    st.set_page_config(
        page_title="AI Sinema Eleştirmeni (Pro)", page_icon="🎬", layout="centered"
    )

SERVICE_DIR = Path(__file__).resolve().parent   # yapay_zeka_servisi klasörü
BASE_DIR = SERVICE_DIR  # eski uyumluluk

# ✅ 3-sınıf dosyalar
TFIDF_BUNDLE_PATH = SERVICE_DIR / "film_tfidf_3cls.pkl"

# ✅ V2 MODEL KLASÖRÜ
BERT_MODEL_PATH = SERVICE_DIR / "benim_bert_modelim_3cls_v2"

print("[OK] SERVICE_DIR:", SERVICE_DIR)
print("[OK] BERT_MODEL_PATH:", BERT_MODEL_PATH, "exists=", BERT_MODEL_PATH.exists())
print("[OK] TFIDF_PATH:", TFIDF_BUNDLE_PATH, "exists=", TFIDF_BUNDLE_PATH.exists())

MAX_TEXT_LENGTH = 5000

# 3-sınıf etiketler (0,1,2) — başlangıç varsayılanları, BERT load sonrası config'ten güncellenir
LABELS = ["OLUMSUZ", "NÖTR", "OLUMLU"]
ID2LABEL = {0: "OLUMSUZ", 1: "NÖTR", 2: "OLUMLU"}
LABEL2ID = {"OLUMSUZ": 0, "NÖTR": 1, "OLUMLU": 2}

# ---------------------------------------------------------------------
# 2) GUARDRAIL + NÖTR KURALLARI
# ---------------------------------------------------------------------
NEG_HINTS = [
    "rezalet", "berbat", "iğrenç", "igrenc", "çöp", "cop", "bok", "kaka",
    "sakın", "sakin", "pişman", "pisman", "fiyasko", "zıkkım", "saçma",
    "boş", "bos", "dandik", "izlenilmez", "kaçın", "kacin", "sıkıcı",
    "sikici", "bayık", "bayik", "uykumu", "vakit_kaybi", "zaman_kaybi",
    "yetersiz", "sığ", "sig", "amatör", "amator", "beceriksiz", "tırt",
    "tirt", "leş", "lez", "kusturucu", "işkence", "iskence", "zulüm",
    "zulum", "katlanılmaz", "ucuz", "basit", "facia", "kepaze", "yavan",
]
POS_HINTS = [
    "mükemmel", "mukemmel", "şaheser", "saheser", "efsane", "bayıldım",
    "harika", "başyapıt", "basyapit", "müthiş", "muthis", "harikulade",
    "tapıyorum", "mutlaka", "soluksuz", "sürükleyici", "surukleyici",
    "masterpiece", "şahane", "sahane", "epik", "kült", "kult", "sarsıcı",
    "vurucu", "derinlikli", "vizyoner", "doyurucu", "fevkalade",
    "kusursuz", "enfes", "şaşırtıcı", "sasirtici", "büyüleyici",
    "buyuleyici", "döktürmüş",
]

NEGATION_TOKENS_RAW = {
    "degil", "değil", "yok", "hic", "hiç", "asla", "katiyen", "maalesef",
    "olmaz", "olamaz", "hicbir", "hiçbir",
}
NEGATION_PHRASES_RAW = {"bile degil", "bile değil"}

NEG_EMOJIS = ["💩", "👎", "🤮", "🤬", "😡", "😠", "🤢", "😴", "🤦", "😤", "📉", "🗑️"]
POS_EMOJIS = ["💯", "🔥", "👍", "❤️", "😍", "🥰", "⭐", "✨", "👏", "🙌", "🤩", "🚀", "🍿", "👑"]

NEG_PHRASES = [
    "sakın gitmeyin", "zaman kaybı", "zaman kaybi", "vakit kaybı", "vakit kaybi",
    "param haram olsun", "haram olsun", "izlemeyin", "izlenmez", "uzak durun",
    "yanına bile yaklaşmayın", "berbat ötesi", "rezalet ötesi", "izlemeye değmez",
    "izlemeye degmez", "pişman oldum", "pisman oldum", "sıkıldım izlerken",
    "sikildim izlerken", "vaktimi çaldı", "vaktimi caldi", "berbat bir film",
    "rezil bir film", "en kötü film", "en kotu film", "tam bir hayal kırıklığı",
    "tam bir hayal kirikligi", "beş para etmez", "bes para etmez", "yarıda bıraktım",
    "yarida biraktim", "sonu saçmaydı", "sonu sacmaydi", "senaryo çok kötü",
    "senaryo cok kotu", "hiç beğenmedim", "hic begenmedim", "hayatımdan çalınan",
    "hayatimdan calinan", "tahammül edemedim", "tahammul edemedim", "içim şişti",
    "icim sisti", "ruhum daraldı", "ruhum daraldi", "gözlerim kanadı", "gozlerim kanadi",
    "beyin yakan saçmalık", "beyin yakan sacmalik", "mantık hatası dolu",
    "mantik hatasi dolu", "oyunculuklar yerlerde", "efektler berbat", "kurgu felaket",
    "tavsiye etmem",
]
POS_PHRASES = [
    "kesinlikle izleyin", "mutlaka izleyin", "kaçırmayın", "kacirmayin", "kaçırma",
    "kacirma", "defalarca izlenir", "herkese tavsiye ederim", "şiddetle tavsiye",
    "siddetle tavsiye", "arşivlik", "arsivlik", "tekrar izleyeceğim", "tekrar izleyecegim",
    "şans verin", "sans verin", "sakın kaçırmayın", "sakin kacirmayin", "favorim oldu",
    "efsane olmuş", "efsane olmus", "muhteşem bir film", "muhtesem bir film",
    "hayran kaldım", "hayran kaldim", "en iyi filmlerden", "favorilerime eklendi",
    "şimdiye kadar izlediğim en iyi", "simdiye kadar izledigim en iyi", "oyunculuk harika",
    "senaryo mükemmel", "senaryo mukemmel", "görüntü yönetmeni döktürmüş",
    "goruntu yonetmeni dokturbus", "oyunculuk resitali", "senaryo çok zekice",
    "senaryo cok zekice", "ters köşe", "ters kose", "sonu mükemmeldi", "sonu mukemmeldi",
    "etkisinden çıkamadım", "etkisinden cikamadim", "ağzımız açık izledik", "agzimiz acik izledik",
    "soluksuz izledim", "gözünü kırpmadan", "gozunu kirpmadan", "su gibi aktı", "su gibi akti",
    "10 numara", "yıldızlı pekiyi", "yildizli pekiyi", "tek kelimeyle muazzam",
    "ayakta alkışlanacak", "ayakta alkislanacak",
]

NEUTRAL_STRICT_PHRASES = [
    "ne iyi ne kotu", "ne cok iyi ne cok kotu", "ne iyi ne de kotu", "ne cok iyi ne de kotu",
    "eh iste", "orta karar",
]
NEUTRAL_SOFT_PHRASES = [
    "ortalama", "vasat", "idare eder", "normal", "soyle boyle", "siradan", "standart",
    "kararsizim", "kotu degil ama iyi de degil", "begenmedim ama kotu degil",
    "mukemmel degil ama kotu de degil", "cok beklentiye girmeyin", "beklentimin altinda",
    "beklentimi karsilamadi", "cerezlik", "vakit gecirmelik", "kafa dagitmalik",
    "yoklukta gider", "pazar sinemasi", "tv filmi tadinda", "bos vakitte izlenir",
    "izlenir ama", "tek seferlik", "bir kere izlenir", "etki yaratmiyor", "akilda kalici degil",
    "iz birakmiyor", "iz birakacak bir etki yaratmiyor", "fikir guzel ama uygulama zayif",
    "potansiyeli harcanmis", "guzel basladi kotu bitti", "iyi basladi ama", "sonu haric",
    "biraz sikici", "fena degildi ama", "guzeldi ama", "abartildigi kadar degil",
    "klise dolu", "siradan bir yapim",
]

CONTRAST_TOKENS = {"ama", "fakat", "ancak", "lakin", "ragmen", "rağmen", "halde"}
CONTRAST_PHRASES = {"yine de", "buna ragmen", "buna rağmen"}

MILD_POS_PHRASES = [
    "fena degil", "fena değil", "fena degildi", "fena değildi", "iyiydi", "guzeldi", "güzeldi",
    "iyi sayilir", "kotu degildi", "kötü değildi", "oyunculuklar iyi", "cekimler guzel",
    "çekimler güzel", "muzikler guzel", "müzikler güzel", "goruntu guzel", "görüntü güzel",
    "fikir guzel", "baslangic iyi", "başlangıç iyi", "atmosfer iyi", "potansiyel var",
    "kurgu iyi", "konu guzel", "mekanlar guzel", "kostumler iyi", "kostümler iyi",
    "efektler iyi", "kotu sayilmaz", "izlenir",
]
MILD_NEG_WORDS = {
    "zayif", "zayıf", "eksik", "sikici", "sıkıcı", "uzun", "yorucu", "vasat", "ortalama",
    "kotu", "kötü", "bayik", "bayık", "sacma", "saçma", "kopuk", "yavas", "yavaş",
    "siradan", "sıradan", "klise", "mantiksiz", "mantıksız", "tutarsiz", "tutarsız",
    "basit", "olmamis", "olmamış", "yapay", "donuk", "abarti", "abartı", "zorlama",
    "tahmin edilebilir", "heyecansiz", "heyecansız", "durgun", "sarkmis", "sarkmış",
    "tempo dusuk", "tempo düşük", "inandirici degil", "inandırıcı değil", "finali kotu",
    "finali kötü",
}

FUZZY_STRICT = 0.90
FUZZY_RELAXED = 0.80

# ✅ Opsiyonel neutral band (kapalı önerilir)
BERT_NEUTRAL_LOW = 0.45
BERT_NEUTRAL_HIGH = 0.65

# ---------------------------------------------------------------------
# SARCASM / IRONY DETECTION
# ---------------------------------------------------------------------
SARCASM_MARKERS = [
    "saka yapiyorum", "saka yapıyorum", "şaka yapıyorum", "şaka yapiyorum",
    "ironi", "ironiydi", "ironiydi ya",
    "tabii ki", "tabi ki", "saka maka",
]

SARCASM_NEG_CUES = [
    "salonu terk", "terk ett", "ciktim", "çıktım", "cikmak",
    "zor tuttum", "dayanamadim", "katlanamadim",
    "berbat", "rezalet", "sıkıcı", "sikici", "iğrenç", "igrenc",
    "vakit kayb", "zaman kayb", "pişman", "pisman", "cop", "çöp",
]


def split_on_sarcasm(text: str):
    """
    Metni ironi belirleyicisinden (marker) böler.
    Returns: (head, tail, marker_found) veya (None, None, None)
    """
    clean = rule_clean(text)
    for m in SARCASM_MARKERS:
        mc = rule_clean(m)
        if mc in clean:
            parts = clean.split(mc, 1)
            head = parts[0].strip() if len(parts) > 0 else ""
            tail = parts[1].strip() if len(parts) > 1 else ""
            return head, tail, m
    return None, None, None


def has_sarcasm_negative_tail(tail: str) -> bool:
    """Tail kısmında açık olumsuz ipuçları var mı kontrol eder."""
    if not tail:
        return False
    return any(cue in tail for cue in SARCASM_NEG_CUES)

_TR_MAP = str.maketrans(
    {
        "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "I": "i", "İ": "i",
        "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
    }
)

def rule_clean(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower().translate(_TR_MAP)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _split_phrases(phrases):
    single = set()
    multi = []
    for ph in phrases:
        ph = (ph or "").strip()
        if not ph:
            continue
        if " " in ph:
            multi.append(ph)
        else:
            single.add(ph)
    return single, multi

def _has_any_phrase(clean: str, toks: list, single_set: set, multi_list: list) -> bool:
    if single_set:
        for t in toks:
            if t in single_set:
                return True
    if multi_list:
        for ph in multi_list:
            if ph in clean:
                return True
    return False

R_NEG_HINTS = [rule_clean(x) for x in NEG_HINTS if rule_clean(x)]
R_POS_HINTS = [rule_clean(x) for x in POS_HINTS if rule_clean(x)]
NEG_SET = set(R_NEG_HINTS)
POS_SET = set(R_POS_HINTS)

R_NEG_PHRASES = [rule_clean(x) for x in NEG_PHRASES if rule_clean(x)]
R_POS_PHRASES = [rule_clean(x) for x in POS_PHRASES if rule_clean(x)]
R_NEUTRAL_STRICT = [rule_clean(x) for x in NEUTRAL_STRICT_PHRASES if rule_clean(x)]
R_NEUTRAL_SOFT = [rule_clean(x) for x in NEUTRAL_SOFT_PHRASES if rule_clean(x)]
R_MILD_POS_PHRASES = [rule_clean(x) for x in MILD_POS_PHRASES if rule_clean(x)]
R_CONTRAST_PHRASES = {rule_clean(x) for x in CONTRAST_PHRASES if rule_clean(x)}

NEG_PH_SINGLE, NEG_PH_MULTI = _split_phrases(R_NEG_PHRASES)
POS_PH_SINGLE, POS_PH_MULTI = _split_phrases(R_POS_PHRASES)
NEU_STR_SINGLE, NEU_STR_MULTI = _split_phrases(R_NEUTRAL_STRICT)
NEU_SFT_SINGLE, NEU_SFT_MULTI = _split_phrases(R_NEUTRAL_SOFT)
MILD_POS_SINGLE, MILD_POS_MULTI = _split_phrases(R_MILD_POS_PHRASES)

NEGATION_TOKENS = {rule_clean(x) for x in NEGATION_TOKENS_RAW if rule_clean(x)}
NEGATION_PHRASES = {rule_clean(x) for x in NEGATION_PHRASES_RAW if rule_clean(x)}

NE_NE_REGEX_1 = re.compile(r"\bne\s+(cok\s+)?iyi\w*\s+ne(\s+(de|da))?\s+(cok\s+)?kotu\w*\b")
NE_NE_REGEX_2 = re.compile(r"\bne\s+(cok\s+)?kotu\w*\s+ne(\s+(de|da))?\s+(cok\s+)?iyi\w*\b")
NE_GENERIC_REGEX = re.compile(r"\bne\b.{0,80}\bne(\s+de)?\b")

NOT_BAD_BUT_RE = re.compile(r"\b(kotu|fena|berbat|rezalet)\b.*?\bdegil\b.*?\b(ama|fakat|ancak|lakin|yine\s+de)\b")
GOOD_BUT_RE = re.compile(
    r"\b(iyi|guzel|harika|basarili|surukleyici|atmosfer|muzik|muzikler|gorsel\w*|oyuncu\w*|efekt\w*)\b.*?\b(ama|fakat|ancak|lakin|yine\s+de|buna\s+ragmen)\b.*?\b(zayif|eksik|sikici|uzun|yavas|kopuk|siradan|vasat|tikan\w*|imkansiz|yoksun|dusuk|dustu|zor|anlamsiz)\b"
)

def negation_near(toks, i, window=2) -> bool:
    left = max(0, i - window)
    right = min(len(toks), i + window + 1)
    window_toks = toks[left:right]
    if any(t in NEGATION_TOKENS for t in window_toks):
        return True
    window_text = " ".join(window_toks)
    if any(ph in window_text for ph in NEGATION_PHRASES):
        return True
    return False

def check_guardrails(text: str, cutoff=0.85):
    """
    Döndürür: "neg" | "pos" | "neutral" | "conflict" | None
    İyileştirme: Conflict kontrolü Nötr için de yapılıyor.
    """
    raw = "" if text is None else str(text)
    neg_e = any(e in raw for e in NEG_EMOJIS)
    pos_e = any(e in raw for e in POS_EMOJIS)
    if neg_e and pos_e:
        return "conflict"
    if neg_e:
        return "neg"
    if pos_e:
        return "pos"

    clean = rule_clean(raw)
    toks = clean.split()

    neu_strict = (
        _has_any_phrase(clean, toks, NEU_STR_SINGLE, NEU_STR_MULTI)
        or bool(NE_NE_REGEX_1.search(clean))
        or bool(NE_NE_REGEX_2.search(clean))
    )
    neu_soft = _has_any_phrase(clean, toks, NEU_SFT_SINGLE, NEU_SFT_MULTI)
    neu_p = neu_strict or neu_soft

    neg_p = _has_any_phrase(clean, toks, NEG_PH_SINGLE, NEG_PH_MULTI)
    pos_p = _has_any_phrase(clean, toks, POS_PH_SINGLE, POS_PH_MULTI)

    hits = int(neu_p) + int(neg_p) + int(pos_p)
    if hits >= 2:
        return "conflict"

    if neg_p:
        return "neg"
    if pos_p:
        return "pos"
    if neu_p:
        return "neutral"

    neg_found = False
    pos_found = False
    for i, w in enumerate(toks):
        if w in NEG_SET and not negation_near(toks, i, window=2):
            neg_found = True
        if w in POS_SET and not negation_near(toks, i, window=2):
            pos_found = True
        if neg_found and pos_found:
            return "conflict"
    if neg_found:
        return "neg"
    if pos_found:
        return "pos"

    strict_thr = max(float(cutoff), FUZZY_STRICT)
    relaxed_thr = min(float(cutoff), FUZZY_RELAXED)
    for i, w in enumerate(toks):
        if len(w) < 4:
            continue
        threshold = strict_thr if len(w) < 6 else relaxed_thr
        for hint in R_NEG_HINTS:
            if difflib.SequenceMatcher(None, w, hint).ratio() >= threshold:
                if not negation_near(toks, i, window=2):
                    neg_found = True
                break
        for hint in R_POS_HINTS:
            if difflib.SequenceMatcher(None, w, hint).ratio() >= threshold:
                if not negation_near(toks, i, window=2):
                    pos_found = True
                break
        if neg_found and pos_found:
            return "conflict"

    if neg_found:
        return "neg"
    if pos_found:
        return "pos"
    return None

def has_soft_neutral_signal(text: str) -> bool:
    clean = rule_clean("" if text is None else str(text))
    toks = clean.split()
    return _has_any_phrase(clean, toks, NEU_SFT_SINGLE, NEU_SFT_MULTI)

def is_neutral_like(text: str, return_reason: bool = False):
    clean = rule_clean("" if text is None else str(text))
    toks = clean.split()
    if _has_any_phrase(clean, toks, NEU_STR_SINGLE, NEU_STR_MULTI):
        return (True, "neutral_strict_phrase") if return_reason else True
    if _has_any_phrase(clean, toks, NEU_SFT_SINGLE, NEU_SFT_MULTI):
        return (True, "soft_neutral_phrase") if return_reason else True
    if NE_GENERIC_REGEX.search(clean) or NE_NE_REGEX_1.search(clean) or NE_NE_REGEX_2.search(clean):
        return (True, "ne_ne") if return_reason else True
    if NOT_BAD_BUT_RE.search(clean):
        return (True, "not_bad_but") if return_reason else True
    if GOOD_BUT_RE.search(clean):
        return (True, "good_but") if return_reason else True

    has_contrast = any(w in toks for w in CONTRAST_TOKENS) or any(ph in clean for ph in R_CONTRAST_PHRASES)
    if not has_contrast:
        return (False, None) if return_reason else False

    pos_hit = _has_any_phrase(clean, toks, MILD_POS_SINGLE, MILD_POS_MULTI) or any(t in toks for t in POS_SET)
    neg_hit = any(t in toks for t in MILD_NEG_WORDS) or any(t in toks for t in NEG_SET)
    ok = bool(pos_hit and neg_hit)
    return (ok, "mixed_pos_neg") if return_reason else ok

def validate_text(text):
    if text is None:
        return None, "Boş metin"
    text = str(text)
    if len(text.strip()) < 2:
        return None, "Boş veya çok kısa metin"
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH], f"Metin {MAX_TEXT_LENGTH} karaktere kısaltıldı"
    return text, None

# ---------------------------------------------------------------------
# 3) TF-IDF & BERT LOADERS
# ---------------------------------------------------------------------
@st.cache_resource
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
        obj = joblib.load(path)
        return _normalize_obj(obj)

    except AttributeError:
        import __main__
        __main__.temizle_liste = temizle_liste
        __main__.temizle_tek = temizle_tek
        try:
            obj = joblib.load(path)
            return _normalize_obj(obj)
        except Exception as e:
            return None, f"TF-IDF bundle yüklenemedi: {e}"

    except Exception as e:
        return None, f"TF-IDF bundle yüklenemedi: {e}"

tfidf_bundle, tfidf_err = load_tfidf_bundle(TFIDF_BUNDLE_PATH)

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
    if found["neg"] is None or found["neutral"] is None or found["pos"] is None:
        found = {"neg": 0, "neutral": 1, "pos": 2}
        assumed = True
    return found, assumed

@st.cache_resource
def load_bert(model_path: Path):
    if not model_path.exists():
        return None, None, None, f"BERT klasörü bulunamadı: {model_path}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
        model.to(device)
        model.eval()

        print("[OK] BERT LOADED num_labels:", model.config.num_labels)
        print("[OK] BERT LOADED id2label:", model.config.id2label)

        # Label map'i config'ten al
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

tokenizer, bert_model, bert_meta, bert_err = load_bert(BERT_MODEL_PATH)

# =============================================
# 🔥 STARTUP SELF-TEST
# =============================================
if tokenizer is not None and bert_model is not None:
    _test_text = "çok ama çok iyi"
    _test_inp = tokenizer(_test_text, return_tensors="pt", truncation=True, max_length=256)
    _test_inp = {k: v.to(bert_meta["device"]) for k, v in _test_inp.items()}
    with torch.no_grad():
        _test_out = bert_model(**_test_inp)
    _test_probs = torch.softmax(_test_out.logits, dim=-1)[0].tolist()
    _test_pred = int(torch.argmax(_test_out.logits, dim=-1).item())
    _LABEL_MAP = {0: "Olumsuz", 1: "Nötr", 2: "Olumlu"}
    print(f"[TEST] SELFTEST: '{_test_text}' probs={[f'{p:.4f}' for p in _test_probs]} pred={_test_pred} label={_LABEL_MAP.get(_test_pred, ID2LABEL.get(_test_pred, _test_pred))}")
else:
    print("[WARN] BERT yüklenemedi, self-test atlandı.")

# ---------------------------------------------------------------------
# ✅ CHUNKING (uzun metinleri parçala)
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
        chunk_ids = ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
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
    texts,
    tokenizer,
    model,
    meta,
    batch_size=16,
    max_length=192,
    chunk_stride=64,
    chunk_mode="mean_max",
):
    """
    ✅ Uzun metinlerde chunking yapar.
    """
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
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=int(max_length),
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
        p = _aggregate_chunk_probs(probs_chunks_all[idxs], mode=str(chunk_mode))
        probs_out[i] = p

    return probs_out

# ---------------------------------------------------------------------
# 5) STATS
# ---------------------------------------------------------------------
_STATS_TEMPLATE = {
    "total": 0,
    "guardrail": 0,
    "neutralrule": 0,
    "uncertainneutral": 0,
    "tfidf": 0,
    "bert": 0,
    "ensemble": 0,
    "error": 0,
}

def _stats_enabled() -> bool:
    return bool(RUNNING_IN_STREAMLIT)

def init_stats():
    if not _stats_enabled():
        return
    if "analysis_stats" not in st.session_state or not isinstance(st.session_state.analysis_stats, dict):
        st.session_state.analysis_stats = dict(_STATS_TEMPLATE)
    else:
        for k, v in _STATS_TEMPLATE.items():
            st.session_state.analysis_stats.setdefault(k, v)

def reset_stats():
    if not _stats_enabled():
        return
    st.session_state.analysis_stats = dict(_STATS_TEMPLATE)

def inc_total():
    if not _stats_enabled():
        return
    init_stats()
    st.session_state.analysis_stats["total"] += 1

def inc_source(source_key: str):
    if not _stats_enabled():
        return
    init_stats()
    key = source_key.lower().replace("-", "").replace(" ", "")
    if key in st.session_state.analysis_stats:
        st.session_state.analysis_stats[key] += 1

def inc_source_n(source_key: str, n: int):
    if not _stats_enabled():
        return
    init_stats()
    key = source_key.lower().replace("-", "").replace(" ", "")
    if key in st.session_state.analysis_stats:
        st.session_state.analysis_stats[key] += int(n)

# ---------------------------------------------------------------------
# 6) ENSEMBLE LOGIC
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
    if tfidf_bundle is None or "model" not in tfidf_bundle:
        raise KeyError("TF-IDF bundle missing model")
    model = tfidf_bundle["model"]
    probs = model.predict_proba(texts)
    if probs.shape[1] == 2:
        out = np.zeros((len(texts), 3), dtype=float)
        out[:, 0] = probs[:, 0]
        out[:, 2] = probs[:, 1]
        return out
    return _reorder_probs_to_012(probs, getattr(model, "classes_", None))

def pick_label_from_probs(p3: np.ndarray):
    idx = int(np.argmax(p3))
    return ID2LABEL[idx], idx, float(p3[idx])

def apply_neutral_band(p3: np.ndarray, low: float, high: float):
    return bool(low <= float(p3[1]) <= high)

def top2_info(p3: np.ndarray):
    order = np.argsort(p3)[::-1]
    top1, top2 = float(p3[order[0]]), float(p3[order[1]])
    return int(order[0]), int(order[1]), top1, top2, top1 - top2

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
    bert_max_len=192,  # ✅ default büyütüldü
    debug_mode=False,
    uncertain_to_neutral_on=False,   # ✅ Kapatıldı — model kararına güveniyoruz
    conf_threshold=0.50,
    margin_threshold=0.15,
    min_neutral_prob=0.25,
):
    logs = []
    try:
        inc_total()
        validated_text, validation_msg = validate_text(text)
        if validated_text is None:
            inc_source("error")
            return "GEÇERSİZ", 0.0, "Error", {"error": validation_msg}

        # --- SARCASM OVERRIDE LOGIC ---
        s_head, s_tail, s_marker = split_on_sarcasm(validated_text)
        if s_tail and len(s_tail) >= 5:
            if debug_mode:
                logs.append(f"Sarcasm marker: '{s_marker}' tail: '{s_tail[:60]}'")
            # Strategy 1: Explicit keyword check on tail
            if has_sarcasm_negative_tail(s_tail):
                inc_source("guardrail")
                return "OLUMSUZ", 0.99, "SarcasmRule", {
                    "marker": s_marker, "reason": "Negative Tail Keywords",
                    "logs": logs if debug_mode else None,
                }
            # Strategy 2: BERT re-evaluation on tail only
            if tokenizer is not None and bert_model is not None:
                try:
                    p_tail = bert_predict_proba_batch(
                        [s_tail], tokenizer, bert_model, bert_meta,
                        batch_size=1, max_length=int(bert_max_len),
                    )[0]
                    tail_label, _, tail_conf = pick_label_from_probs(p_tail)
                    if tail_label == "OLUMSUZ" and tail_conf >= 0.60:
                        inc_source("guardrail")
                        return "OLUMSUZ", float(tail_conf), "Sarcasm->BERTTail", {
                            "marker": s_marker, "tail_text": s_tail[:50],
                            "logs": logs if debug_mode else None,
                        }
                except Exception as e:
                    if debug_mode:
                        logs.append(f"Sarcasm BERT check failed: {e}")
        # --- SARCASM OVERRIDE LOGIC END ---

        if use_guardrail:
            hint = check_guardrails(validated_text, cutoff=guard_cutoff)
            if debug_mode:
                logs.append(f"Guardrail hint: {hint}")
            if hint == "neutral":
                inc_source("guardrail")
                return "NÖTR", 0.99, "Guardrail", {"hint": "neutral", "logs": logs if debug_mode else None}
            if hint == "neg":
                inc_source("guardrail")
                return "OLUMSUZ", 0.99, "Guardrail", {"hint": "neg", "logs": logs if debug_mode else None}
            if hint == "pos":
                inc_source("guardrail")
                return "OLUMLU", 0.99, "Guardrail", {"hint": "pos", "logs": logs if debug_mode else None}

        if neutral_on:
            is_neu, reason = is_neutral_like(validated_text, return_reason=True)
            if debug_mode:
                logs.append(f"NeutralRule: {is_neu} ({reason})")
            if is_neu:
                inc_source("neutralrule")
                return "NÖTR", 0.99, "NeutralRule", {"reason": reason, "logs": logs if debug_mode else None}

        p_tfidf = tfidf_predict_proba([validated_text])[0]
        inc_source("tfidf")
        if debug_mode:
            logs.append(f"TFIDF: N={p_tfidf[0]:.2f} U={p_tfidf[1]:.2f} P={p_tfidf[2]:.2f}")

        if tokenizer is None or bert_model is None:
            raise RuntimeError("BERT failed")

        p_bert = bert_predict_proba_batch(
            [validated_text],
            tokenizer,
            bert_model,
            bert_meta,
            batch_size=1,
            max_length=int(bert_max_len),
        )[0]
        inc_source("bert")
        if debug_mode:
            logs.append(f"BERT : N={p_bert[0]:.2f} U={p_bert[1]:.2f} P={p_bert[2]:.2f}")

        if use_neutral_band and apply_neutral_band(p_bert, bert_neutral_low, bert_neutral_high):
            inc_source("ensemble")
            return "NÖTR", float(p_bert[1]), "Neutral-BERTBand", {"p_bert": p_bert.tolist(), "logs": logs if debug_mode else None}

        tw, bw = float(tfidf_weight), float(bert_weight)
        if tw + bw <= 0:
            tw, bw = 0.5, 0.5
        p_mix = tw * p_tfidf + bw * p_bert
        p_mix /= p_mix.sum() + 1e-12

        if debug_mode:
            logs.append(f"MIX  : N={p_mix[0]:.2f} U={p_mix[1]:.2f} P={p_mix[2]:.2f} (Weights T={tw} B={bw})")

        if uncertain_to_neutral_on:
            _, _, top1, top2, margin = top2_info(p_mix)
            p_neu = float(p_mix[1])
            soft_neu = has_soft_neutral_signal(validated_text)

            conf_eff = max(float(conf_threshold), 0.57) if soft_neu else float(conf_threshold)
            margin_eff = max(float(margin_threshold), 0.22) if soft_neu else float(margin_threshold)
            min_neu_eff = min(float(min_neutral_prob), 0.18) if soft_neu else float(min_neutral_prob)

            is_uncertain = (top1 < conf_eff) and (margin < margin_eff) and (p_neu >= min_neu_eff)

            if debug_mode:
                logs.append(f"Uncertainty: Top1={top1:.2f} Margin={margin:.2f} P_Neu={p_neu:.2f} | SoftSig={soft_neu}")
                logs.append(f"-> Thresholds: Conf<{conf_eff:.2f} Margin<{margin_eff:.2f} Neu>={min_neu_eff:.2f} => IS_UNCERTAIN={is_uncertain}")

            if is_uncertain:
                inc_source("uncertainneutral")
                return "NÖTR", p_neu, "Uncertain→Neutral", {"logs": logs if debug_mode else None}

        label, idx, conf = pick_label_from_probs(p_mix)
        inc_source("ensemble")
        return label, conf, "Ensemble", {"p_mix": p_mix.tolist(), "logs": logs if debug_mode else None}

    except Exception as e:
        inc_source("error")
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
    bert_max_len=192,  # ✅ default büyütüldü
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
            inc_total()
            vt, msg = validate_text(texts[i])
            if vt is None:
                labels[i] = "GEÇERSİZ"
                conf_scores[i] = 0.0
                sources[i] = "Error"
                unresolved[i] = False
                inc_source("error")
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
                    inc_source("guardrail")

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
                    inc_source("neutralrule")

        if progress_callback:
            progress_callback(0.35)

        idxs = np.where(unresolved)[0]
        if len(idxs) > 0:
            tfidf_texts = [processed[i] for i in idxs]
            p_tfidf_all = tfidf_predict_proba(tfidf_texts)
            inc_source_n("tfidf", len(idxs))

            bert_texts = [processed[i] for i in idxs]
            p_bert_all = bert_predict_proba_batch(
                bert_texts,
                tokenizer,
                bert_model,
                bert_meta,
                batch_size=max(1, int(bert_batch_size)),
                max_length=int(bert_max_len),
            )
            inc_source_n("bert", len(idxs))

            tw, bw = float(tfidf_weight), float(bert_weight)
            if tw + bw <= 0:
                tw, bw = 0.5, 0.5

            for k, i in enumerate(idxs):
                p_tfidf = p_tfidf_all[k]
                p_bert = p_bert_all[k]

                if use_neutral_band and apply_neutral_band(p_bert, bert_neutral_low, bert_neutral_high):
                    labels[i] = "NÖTR"
                    conf_scores[i] = float(p_bert[1])
                    sources[i] = "Neutral-BERTBand"
                    unresolved[i] = False
                    inc_source("ensemble")
                    continue

                p_mix = tw * p_tfidf + bw * p_bert
                p_mix = p_mix / (p_mix.sum() + 1e-12)

                if uncertain_to_neutral_on:
                    _, _, t1, t2, marg = top2_info(p_mix)
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
                        inc_source("uncertainneutral")
                        continue

                lab, _, conf = pick_label_from_probs(p_mix)
                labels[i] = lab
                conf_scores[i] = conf
                sources[i] = "Ensemble"
                unresolved[i] = False
                inc_source("ensemble")

        if progress_callback:
            progress_callback(1.0)

        return labels.tolist(), conf_scores, sources.tolist()

    except Exception as e:
        if RUNNING_IN_STREAMLIT:
            st.error(f"Toplu analiz hatası: {e}")
        return ["HATA"] * len(texts), np.zeros(len(texts)), ["Error"] * len(texts)

# ---------------------------------------------------------------------
# 7) UI
# ---------------------------------------------------------------------
if RUNNING_IN_STREAMLIT:
    init_stats()
    st.title("🎬 AI Sinema Eleştirmeni (Ensemble Pro)")
    st.caption("3-Sınıf (Neg/Neu/Pos) | BERT ağırlıklı Ensemble | Gelişmiş Belirsizlik Yönetimi")
    st.divider()

    if tfidf_err:
        st.error(f"❌ {tfidf_err}")
        st.stop()
    if bert_err:
        st.error(f"❌ {bert_err}")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Ayarlar")
        st.success(f"Cihaz: {bert_meta['device'].upper()}")

        use_guardrail = st.toggle("Guardrails aktif", value=True)
        guard_cutoff = st.slider("Guardrail fuzzy cutoff", 0.70, 0.95, 0.85, 0.01)

        st.divider()
        st.subheader("Ensemble Ağırlıkları")
        tfidf_weight = st.slider("TF-IDF Ağırlığı", 0.0, 1.0, 0.30, 0.05)
        bert_weight = 1.0 - tfidf_weight
        st.caption(f"BERT Ağırlığı: {bert_weight:.2f}")

        st.divider()
        st.subheader("Belirsizlik Yönetimi")
        uncertain_to_neutral_on = st.toggle("Belirsizse NÖTR'e çek", value=True)
        conf_threshold = st.slider("Güven Eşiği (Max < x)", 0.30, 0.80, 0.50, 0.05)
        margin_threshold = st.slider("Fark Eşiği (Margin < x)", 0.05, 0.30, 0.15, 0.01)
        min_neutral_prob = st.slider("Min Nötr Olasılığı", 0.10, 0.50, 0.25, 0.05)
        st.info("Eğer model emin değilse ve metinde 'ortalama' vb. sinyaller varsa karar NÖTR olur.")

        st.divider()
        neutral_on = st.toggle("Nötr Kuralı (Regex/Phrase)", value=False)
        use_neutral_band = st.toggle("BERT Bandı (Opsiyonel)", value=False)
        bert_neutral_low = 0.45
        bert_neutral_high = 0.65

        st.divider()
        bert_batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=1)
        bert_max_len = st.selectbox("Max length", [96, 128, 160, 192, 256], index=3)  # ✅ default 192
        debug_mode = st.checkbox("Debug Modu (Tekli)", value=False)

        if st.button("🧹 İstatistikleri Sıfırla"):
            reset_stats()
            st.success("Sıfırlandı.")

        st.divider()
        st.subheader("📊 İstatistikler")
        stats = st.session_state.analysis_stats
        if stats["total"] > 0:
            st.metric("Toplam", stats["total"])
            st.write(f"🛡️ Guard: {stats['guardrail']}")
            st.write(f"🟦 TF-IDF: {stats['tfidf']}")
            st.write(f"🟪 BERT: {stats['bert']}")
            st.write(f"🤷 Uncertain→Nötr: {stats['uncertainneutral']}")
            st.write(f"🧩 Ensemble: {stats['ensemble']}")
            st.write(f"❌ Error: {stats['error']}")

    tab1, tab2 = st.tabs(["💬 Tekli Analiz", "📂 Toplu Analiz"])

    with tab1:
        if "single_text" not in st.session_state:
            st.session_state["single_text"] = ""

        def clear_single_text():
            st.session_state["single_text"] = ""

        c1, c2 = st.columns([1, 4])
        yorum = st.text_area("Yorum:", height=110, key="single_text")

        if c1.button("ANALİZ ET 🚀", key="btn_single"):
            if len(str(yorum).strip()) < 2:
                st.warning("Yorum girin.")
            else:
                with st.spinner("Analiz ediliyor..."):
                    label, conf, src, dbg = ensemble_single(
                        yorum,
                        use_guardrail,
                        guard_cutoff,
                        neutral_on,
                        use_neutral_band,
                        bert_neutral_low,
                        bert_neutral_high,
                        tfidf_weight,
                        bert_weight,
                        bert_batch_size,
                        bert_max_len,
                        debug_mode,
                        uncertain_to_neutral_on,
                        conf_threshold,
                        margin_threshold,
                        min_neutral_prob,
                    )
                st.subheader("Sonuç")
                color = "green" if label == "OLUMLU" else "red" if label == "OLUMSUZ" else "orange"
                st.markdown(f"### :{color}[{label}]")
                st.progress(int(conf * 100))
                st.info(f"Güven: %{conf*100:.1f} | Kaynak: {src}")
                with st.expander("Detaylar & Debug"):
                    st.write(dbg)

        c2.button("TEMİZLE", on_click=clear_single_text)

    with tab2:
        st.info("CSV/Excel yükleyin. 'yorum' sütunu aranır.")
        up = st.file_uploader("Dosya Yükle", type=["csv", "xlsx"], key="file_uploader")
        if up:
            try:
                df = pd.read_csv(up, on_bad_lines="skip") if up.name.endswith(".csv") else pd.read_excel(up)
                cols = [c for c in df.columns if any(x in str(c).lower() for x in ["yorum", "text", "review"])]
                target_col = st.selectbox(
                    "Sütun Seç:",
                    df.columns.tolist(),
                    index=df.columns.tolist().index(cols[0]) if cols else 0,
                )

                if st.button("Başlat"):
                    texts = df[target_col].fillna("").astype(str).tolist()

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(p):
                        progress_bar.progress(int(p * 100))
                        status_text.text(f"%{int(p*100)}")

                    with st.spinner("Çalışıyor..."):
                        labels, confs, srcs = ensemble_batch(
                            texts,
                            use_guardrail,
                            guard_cutoff,
                            neutral_on,
                            use_neutral_band,
                            bert_neutral_low,
                            bert_neutral_high,
                            tfidf_weight,
                            bert_weight,
                            bert_batch_size,
                            bert_max_len,
                            update_progress,
                            uncertain_to_neutral_on,
                            conf_threshold,
                            margin_threshold,
                            min_neutral_prob,
                        )

                    progress_bar.empty()
                    status_text.empty()

                    df["AI_Karari"] = labels
                    df["Guven_%"] = np.round(np.array(confs) * 100, 1)
                    df["Kaynak"] = srcs
                    st.success("Bitti!")

                    counts = df["AI_Karari"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Olumlu", counts.get("OLUMLU", 0))
                    c2.metric("Olumsuz", counts.get("OLUMSUZ", 0))
                    c3.metric("Nötr", counts.get("NÖTR", 0))

                    st.dataframe(df.head(20))
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("İndir (CSV)", csv, "sonuc.csv", "text/csv")

                    if px:
                        fig = px.pie(values=counts.values, names=counts.index, title="Dağılım")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Hata: {e}")

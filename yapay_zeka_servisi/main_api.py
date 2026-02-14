import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app_ensemble import ensemble_single, tfidf_err, bert_err

# UI'ye uygun label normalizasyonu
LABEL_UI = {"OLUMLU": "Olumlu", "OLUMSUZ": "Olumsuz", "NÖTR": "Nötr"}

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Sinema Eleştirmeni API",
    description="Film yorumlarını BERT + TF-IDF ensemble ile analiz eder.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"ok": True, "service": "ai-sinema", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "ok": (not tfidf_err and not bert_err),
        "tfidf_err": tfidf_err,
        "bert_err": bert_err,
        "models": {
            "tfidf": "loaded" if not tfidf_err else "error",
            "bert": "loaded" if not bert_err else "error",
        },
    }


@app.post("/analiz")
async def analiz(request: Request):
    """
    Yorum metnini alır, duygu analizi yapar.
    Body: {"yorum_metni": "..."}
    Content-Type: application/json olmalı.
    """
    # ---- 1) Body'yi güvenli şekilde parse et ----
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        logger.error("JSON parse hatası. Raw body: %s", raw[:200])
        raise HTTPException(
            status_code=422,
            detail=(
                "JSON parse hatası. İsteğin body kısmı geçerli bir JSON olmalı "
                "ve Content-Type: application/json header'ı gerekli. "
                f"Gelen ham body: {raw[:200]}"
            ),
        )

    # ---- 2) yorum_metni alanını al ----
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body bir JSON objesi olmalı, örn: {\"yorum_metni\": \"...\"}")

    text = (body.get("yorum_metni") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="yorum_metni boş olamaz.")
    if len(text) > 20000:
        raise HTTPException(status_code=413, detail="yorum_metni çok uzun (max 20000 karakter).")

    # ---- 3) Model kontrolü ----
    if tfidf_err or bert_err:
        logger.error("Model yükleme hatası — tfidf: %s, bert: %s", tfidf_err, bert_err)
        raise HTTPException(status_code=503, detail={"tfidf_err": tfidf_err, "bert_err": bert_err})

    # ---- 4) Analiz ----
    try:
        label, conf, src, dbg = ensemble_single(text)
    except Exception as e:
        logger.exception("ensemble_single çalışırken hata: %s", e)
        raise HTTPException(status_code=500, detail="Model analiz sırasında hata oluştu.")

    if label == "HATA":
        logger.warning("Analiz HATA döndü: %s", dbg)
        raise HTTPException(status_code=500, detail=dbg)

    logger.info("Analiz: '%s...' → %s (%.4f) [%s]", text[:60], label, float(conf), src)

    return {
        "karar": LABEL_UI.get(label, label),
        "guven_skoru": float(conf),
        "kaynak": src,
        "debug": dbg,
    }

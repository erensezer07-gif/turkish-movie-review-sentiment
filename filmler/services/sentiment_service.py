import logging

try:
    from sinema_sitesi.ai_client import analiz_yap
except ImportError:
    analiz_yap = None
except Exception as e:
    logging.getLogger(__name__).error(f"AI Client import hatası: {e}")
    analiz_yap = None

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"OLUMLU", "OLUMSUZ", "NÖTR"}


def _normalize_decision(value) -> str:
    """Ensemble her zaman OLUMLU/OLUMSUZ/NÖTR/HATA döndürür;
    yine de güvenlik için case/whitespace toleranslı eşle.
    """
    if not value:
        return "NÖTR"
    norm = str(value).strip().upper().replace("Ö", "Ö")  # idempotent
    if norm in VALID_DECISIONS:
        return norm
    # Esnek eşleşmeler (eski API uyumu)
    if norm.startswith("OLUMLU"):
        return "OLUMLU"
    if norm.startswith("OLUMSUZ"):
        return "OLUMSUZ"
    if norm.startswith("KARARSIZ") or norm.startswith("NÖTR") or norm == "NOTR":
        return "NÖTR"
    return "NÖTR"


def analyze_comment(text: str):
    """
    Yorum metnini analiz eder ve sonucu döner.
    Dönüş: {decision, confidence, source, duration}
    """
    if not text or not isinstance(text, str):
        return {"decision": "NÖTR", "confidence": 0.0, "source": "invalid_input", "duration": 0.0}

    if analiz_yap is None:
        logger.warning("AI client (analiz_yap) yüklenmedi.")
        return {"decision": "NÖTR", "confidence": 0.0, "source": "api_error", "duration": 0.0}

    try:
        sonuc = analiz_yap(text)
    except Exception as e:
        logger.warning("AI servisi hatası: %s", e)
        return {"decision": "NÖTR", "confidence": 0.0, "source": "api_error", "duration": 0.0}

    decision = _normalize_decision(sonuc.get("karar"))
    confidence = float(sonuc.get("guven_skoru", 0.0))
    source = sonuc.get("kaynak")
    duration = float(sonuc.get("sure_sn", 0.0))

    logger.info("Sentiment Service: %s (%.4fs) [%s]", decision, duration, source)

    return {"decision": decision, "confidence": confidence, "source": source, "duration": duration}


def get_sentiment_badge(decision: str):
    """Karara göre UI badge bilgileri."""
    badges = {
        "OLUMLU": {"text": "Olumlu", "cls": "badge bg-success text-white p-2", "sent": "pos"},
        "OLUMSUZ": {"text": "Olumsuz", "cls": "badge bg-danger text-white p-2", "sent": "neg"},
    }
    return badges.get(decision, {"text": "Nötr", "cls": "badge bg-secondary text-white p-2", "sent": "neu"})

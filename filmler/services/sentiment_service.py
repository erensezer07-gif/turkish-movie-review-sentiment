import logging
from django.conf import settings

# ✅ AI client (Django -> FastAPI)
# Dosya yoksa oluştur: sinema_sitesi/ai_client.py
try:
    from sinema_sitesi.ai_client import analiz_yap
except Exception:
    analiz_yap = None

logger = logging.getLogger(__name__)

AI_API_URL = getattr(settings, "AI_API_URL", "http://127.0.0.1:8001")


def analyze_comment(text: str):
    """
    Yorum metnini analiz eder ve sonucu döner.
    Dönen sözlük: {
        'decision': 'OLUMLU' | 'OLUMSUZ' | 'NÖTR',
        'confidence': float,
        'source': str
    }
    """
    # Default (AI servis yoksa bile yorum kaydolur)
    ai_karar = "NÖTR"
    ai_guven = 0.0
    ai_kaynak = None

    # API label → DB format normalizasyonu
    KARAR_NORMALIZE = {
        "Olumlu": "OLUMLU", "olumlu": "OLUMLU", "OLUMLU": "OLUMLU",
        "Olumsuz": "OLUMSUZ", "olumsuz": "OLUMSUZ", "OLUMSUZ": "OLUMSUZ",
        "Kararsız": "NÖTR", "kararsız": "NÖTR", "NÖTR": "NÖTR", "Nötr": "NÖTR",
    }

    # AI Analiz
    ai_sure = 0.0
    try:
        if analiz_yap is None:
            raise RuntimeError("ai_client import edilemedi.")
        sonuc = analiz_yap(text)
        raw_karar = sonuc.get("karar", "NÖTR")
        ai_karar = KARAR_NORMALIZE.get(raw_karar, "NÖTR")
        ai_guven = float(sonuc.get("guven_skoru", 0.0))
        ai_kaynak = sonuc.get("kaynak")
        ai_sure = float(sonuc.get("sure_sn", 0.0))
        
        # Service seviyesinde de loglayalim (isteğe bağlı, ai_client zaten logluyor ama burada da dursun)
        logger.info("Sentiment Service: %s (%.4fs) [%s]", ai_karar, ai_sure, ai_kaynak)
        
    except Exception as e:
        logger.warning("AI servisine bağlanılamadı: %s | AI_API_URL=%s", e, AI_API_URL)
        ai_kaynak = "api_error"

    return {
        "decision": ai_karar,
        "confidence": ai_guven,
        "source": ai_kaynak,
        "duration": ai_sure
    }


def get_sentiment_badge(decision: str):
    """
    Karara (decision) göre UI badge bilgilerini döner.
    """
    BADGE_MAP = {
        "OLUMLU": {"text": "Olumlu", "cls": "badge bg-success text-white p-2", "sent": "pos"},
        "OLUMSUZ": {"text": "Olumsuz", "cls": "badge bg-danger text-white p-2", "sent": "neg"},
    }
    return BADGE_MAP.get(decision, {"text": "Nötr", "cls": "badge bg-secondary text-white p-2", "sent": "neu"})

import logging
from django.conf import settings

# ✅# AI client (Django -> FastAPI)
# Dosya yoksa oluştur: sinema_sitesi/ai_client.py
try:
    from sinema_sitesi.ai_client import analiz_yap
except ImportError:
    analiz_yap = None
except Exception as e:
    # ImportError dışında bir hata varsa (SyntaxError vb.) loglayalım ama servisi çökertmeyelim
    logging.getLogger(__name__).error(f"AI Client import hatası: {e}")
    analiz_yap = None

logger = logging.getLogger(__name__)

AI_API_URL = getattr(settings, "AI_API_URL", "http://127.0.0.1:8001")

# API label → DB format normalizasyonu
KARAR_NORMALIZE = {
    "Olumlu": "OLUMLU", "olumlu": "OLUMLU", "OLUMLU": "OLUMLU",
    "Olumsuz": "OLUMSUZ", "olumsuz": "OLUMSUZ", "OLUMSUZ": "OLUMSUZ",
    "Kararsız": "NÖTR", "kararsız": "NÖTR", "NÖTR": "NÖTR", "Nötr": "NÖTR",
}


def analyze_comment(text: str):
    """
    Yorum metnini analiz eder ve sonucu döner.
    Dönen sözlük: {
        'decision': 'OLUMLU' | 'OLUMSUZ' | 'NÖTR',
        'confidence': float,
        'source': str
    }
    """
    # 1. Input Validation
    if not text or not isinstance(text, str):
        return {
            "decision": "NÖTR",
            "confidence": 0.0,
            "source": "invalid_input",
            "duration": 0.0
        }

    # 2. AI Call
    try:
        if analiz_yap is None:
            raise RuntimeError("AI client (analiz_yap) yüklenemedi.")
        
        sonuc = analiz_yap(text)
        
    except Exception as e:
        logger.warning("AI servisine bağlanılamadı: %s | AI_API_URL=%s", e, AI_API_URL)
        return {
            "decision": "NÖTR",
            "confidence": 0.0,
            "source": "api_error",
            "duration": 0.0
        }

    # 3. Normalization
    raw_karar = sonuc.get("karar", "NÖTR")
    ai_karar = KARAR_NORMALIZE.get(raw_karar, "NÖTR")
    ai_guven = float(sonuc.get("guven_skoru", 0.0))
    ai_kaynak = sonuc.get("kaynak")
    ai_sure = float(sonuc.get("sure_sn", 0.0))

    # Log
    logger.info("Sentiment Service: %s (%.4fs) [%s]", ai_karar, ai_sure, ai_kaynak)

    # 4. Return
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

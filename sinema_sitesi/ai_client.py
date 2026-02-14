import requests
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Direct mode için global model değişkeni (lazy load için)
_ensemble_module = None

def get_ensemble_module():
    global _ensemble_module
    if _ensemble_module is None:
        try:
            # sys.path ayarı gerekebilir, ancak proje kök dizini genelde path'tedir.
            from yapay_zeka_servisi import app_ensemble
            _ensemble_module = app_ensemble
            logger.info("Yapay Zeka modulu basariyla yuklendi (Direct Mode).")
        except ImportError as e:
            logger.error(f"Yapay Zeka modulu yuklenemedi: {e}")
            return None
    return _ensemble_module

def analiz_yap(yorum_metni: str) -> dict:
    """
    AI servisine yorum metnini gönderir veya doğrudan analiz yapar.
    Mod: settings.AI_MODE ('direct' veya 'api')
    """
    mode = getattr(settings, "AI_MODE", "direct")
    
    # --- DIRECT MODE ---
    if mode == "direct":
        mod = get_ensemble_module()
        if mod:
            try:
                # ensemble_single fonksiyonunu çağır
                # Dönüş: label, conf, src, dbg
                label, conf, src, dbg = mod.ensemble_single(yorum_metni)
                
                # Hata durumu kontrolü
                if label == "HATA":
                    return {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "error", "debug": dbg}

                # UI için label normalizasyonu (app_ensemble ui uyumlu dönmüyor olabilir, kontrol et)
                # app_ensemble "OLUMLU", "OLUMSUZ", "NÖTR" dönüyor.
                # Views.py zaten bunu mapliyor ama biz yine de standart yapalım.
                return {
                    "karar": label,
                    "guven_skoru": float(conf),
                    "kaynak": f"local::{src}",
                    "debug": dbg
                }
            except Exception as e:
                logger.exception("Direct analiz hatası: %s", e)
                # Hata durumunda API'ye fallback yapılabilir ama şimdilik hata dönelim
                return {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "exception"}
        else:
            logger.warning("Direct mode seçili ama modül yüklenemedi. API deneniyor...")
    
    # --- API MODE (Fallback) ---
    base_url = settings.AI_API_URL.rstrip("/")
    url = f"{base_url}/analiz"
    timeout = getattr(settings, "AI_API_TIMEOUT", 10)

    try:
        r = requests.post(url, json={"yorum_metni": yorum_metni}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.error("AI API hatası: %s", e)
        return {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "api_error"}
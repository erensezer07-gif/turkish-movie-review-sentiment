import requests
import logging
import time
import threading
from django.conf import settings

logger = logging.getLogger(__name__)

# Direct mode için global model değişkeni (lazy load için)
_ensemble_module = None
_model_loading_lock = threading.Lock()

def load_model():
    """
    Modeli yükler (eğer henüz yüklenmemişse).
    Genelde uygulama başlangıcında (apps.py ready()) çağrılır.
    """
    global _ensemble_module
    if _ensemble_module is not None:
        return _ensemble_module

    with _model_loading_lock:
        if _ensemble_module is None:
            try:
                logger.info("Yapay Zeka modulu yukleniyor (Warmup)...")
                # sys.path ayarı gerekebilir, ancak proje kök dizini genelde path'tedir.
                from yapay_zeka_servisi import app_ensemble
                _ensemble_module = app_ensemble
                logger.info("Yapay Zeka modulu basariyla yuklendi (Direct Mode).")
            except ImportError as e:
                logger.error(f"Yapay Zeka modulu yuklenemedi: {e}")
                return None
    return _ensemble_module

def get_ensemble_module():
    """Global model değişkenini döner, yoksa yüklemeyi dener."""
    if _ensemble_module is None:
        return load_model()
    return _ensemble_module

def analiz_yap(yorum_metni: str) -> dict:
    """
    AI servisine yorum metnini gönderir veya doğrudan analiz yapar.
    Mod: settings.AI_MODE ('direct' veya 'api')
    """
    start_time = time.time()
    mode = getattr(settings, "AI_MODE", "direct")
    result = {}
    
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
                    result = {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "error", "debug": dbg}
                else:
                    # UI için label normalizasyonu (app_ensemble ui uyumlu dönmüyor olabilir, kontrol et)
                    # app_ensemble "OLUMLU", "OLUMSUZ", "NÖTR" dönüyor.
                    # Views.py zaten bunu mapliyor ama biz yine de standart yapalım.
                    result = {
                        "karar": label,
                        "guven_skoru": float(conf),
                        "kaynak": f"local::{src}",
                        "debug": dbg
                    }
            except Exception as e:
                logger.exception("Direct analiz hatası: %s", e)
                # Hata durumunda API'ye fallback yapılabilir ama şimdilik hata dönelim
                result = {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "exception"}
        else:
            logger.warning("Direct mode seçili ama modül yüklenemedi. API deneniyor...")
            # Fallback to API logic below if module failed to load
            pass

    # --- API MODE (Fallback if direct failed or mode is api) ---
    if not result:
        base_url = settings.AI_API_URL.rstrip("/")
        url = f"{base_url}/analiz"
        timeout = getattr(settings, "AI_API_TIMEOUT", 10)

        try:
            r = requests.post(url, json={"yorum_metni": yorum_metni}, timeout=timeout)
            r.raise_for_status()
            result = r.json()
            # API'den gelen kaynak bilgisini koru veya ekle
            if "kaynak" not in result:
                result["kaynak"] = "api"
        except requests.RequestException as e:
            logger.error("AI API hatası: %s", e)
            result = {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "api_error"}

    # Zamanlama
    duration = time.time() - start_time
    result["sure_sn"] = duration
    
    # Loglama
    logger.info(
        "Analiz tamamlandı | Mod: %s | Kaynak: %s | Karar: %s | Süre: %.4fs",
        mode,
        result.get("kaynak"),
        result.get("karar"),
        duration
    )

    return result
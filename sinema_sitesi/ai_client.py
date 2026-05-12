import logging
import time
import threading

logger = logging.getLogger(__name__)

_ensemble_module = None
_model_loading_lock = threading.Lock()


def load_model():
    """
    Ensemble modülünü tek seferde yükler (thread-safe).
    `apps.py.ready()` warmup'tan veya ilk istek üzerinden çağrılır.
    """
    global _ensemble_module
    if _ensemble_module is not None:
        return _ensemble_module

    with _model_loading_lock:
        if _ensemble_module is None:
            try:
                logger.info("Yapay Zeka modulu yukleniyor (Warmup)...")
                from yapay_zeka_servisi import app_ensemble
                _ensemble_module = app_ensemble
                logger.info("Yapay Zeka modulu basariyla yuklendi.")
            except ImportError as e:
                logger.error(f"Yapay Zeka modulu yuklenemedi: {e}")
                return None
    return _ensemble_module


def get_ensemble_module():
    if _ensemble_module is None:
        return load_model()
    return _ensemble_module


def analiz_yap(yorum_metni: str) -> dict:
    """
    Yorumu in-process ensemble ile analiz eder.
    Dönüş: {"karar", "guven_skoru", "kaynak", "sure_sn", "debug"}
    """
    start_time = time.time()
    mod = get_ensemble_module()

    if mod is None:
        result = {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "module_unavailable"}
    else:
        try:
            label, conf, src, dbg = mod.ensemble_single(yorum_metni)
            if label == "HATA":
                result = {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "error", "debug": dbg}
            else:
                result = {
                    "karar": label,
                    "guven_skoru": float(conf),
                    "kaynak": f"local::{src}",
                    "debug": dbg,
                }
        except Exception as e:
            logger.exception("Direct analiz hatasi: %s", e)
            result = {"karar": "NÖTR", "guven_skoru": 0.0, "kaynak": "exception"}

    result["sure_sn"] = time.time() - start_time
    logger.info(
        "Analiz tamamlandi | Kaynak: %s | Karar: %s | Sure: %.4fs",
        result.get("kaynak"),
        result.get("karar"),
        result["sure_sn"],
    )
    return result

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

try:
    from yapay_zeka_servisi.app_ensemble import ensemble_single
except ImportError:
    # Lokal calistirmada path sorunu olursa
    from app_ensemble import ensemble_single

# Loglama
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sezer Film AI API", version="1.0")

class YorumModel(BaseModel):
    yorum_metni: str

@app.get("/")
def read_root():
    return {"durum": "aktif", "servis": "Sezer Film AI"}

@app.post("/analiz")
def analiz_et(veri: YorumModel):
    try:
        label, conf, src, dbg = ensemble_single(veri.yorum_metni)
        
        # Hata kontrolü
        if label == "HATA":
            logger.error(f"Analiz hatası: {dbg}")
            return {
                "karar": "NÖTR",
                "guven_skoru": 0.0,
                "kaynak": "error",
                "debug": dbg
            }

        return {
            "karar": label,
            "guven_skoru": float(conf),
            "kaynak": f"api::{src}",
            "debug": dbg
        }
    except Exception as e:
        logger.exception("API Analiz sırasında hata: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

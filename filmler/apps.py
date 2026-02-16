from django.apps import AppConfig
import threading
import logging
import time

logger = logging.getLogger(__name__)

class FilmlerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "filmler"
    verbose_name = "Film Yönetimi"

    def ready(self):
        # Uygulama ayağa kalktığında modeli arka planda yükle (Warmup)
        # Bu sayede ilk istekte bekleme süresi azalır.
        
        # Sadece runserver veya production server çalışırken yapmalı,
        # migrate vs. komutlarında gereksiz yüklemeyi önlemek için kontrol eklenebilir.
        # Basitlik adına threading ile exception handling ekleyerek çağırıyoruz.
        
        def warmup():
            # Döngüsel importu önlemek için içeride import
            try:
                # 2 saniye bekle ki Django tamamen yüklensin
                time.sleep(2)
                from sinema_sitesi.ai_client import load_model
                load_model()
            except Exception as e:
                logger.warning(f"Model warmup hatası: {e}")

        # Daemon thread: Ana program kapanınca bu da kapansın
        t = threading.Thread(target=warmup, daemon=True)
        t.start()

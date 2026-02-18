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
        
        # 1. Kontrollü Preload (Warmup)
        # Yönetim komutlari (migrate, makemigrations, shell, check vb.) calisirken
        # modelin yuklenmesini engelliyoruz.
        
        import sys
        import os

        # Hangi komutlarda preload iptal edilecek?
        SKIP_COMMANDS = {
            'migrate', 'makemigrations', 'collectstatic',
            'test', 'check', 'shell', 'dbshell', 'flush',
            'showmigrations', 'sqlmigrate'
        }
        
        # sys.argv[1] genelde komut ismidir (manage.py runserver ...)
        # Ancak production'da (gunicorn/uvicorn) sys.argv farklı olabilir.
        if len(sys.argv) > 1:
            command = sys.argv[1]
            if command in SKIP_COMMANDS:
                logger.info(f"AI Warmup SKIPPED (Command: {command})")
                return

        # 2. Env Var ile Kontrol (ENABLE_AI_WARMUP=False veya DISABLE_WARMUP=1 ise yapma)
        # Default: True (Production'da workers icin)
        enable_warmup = os.environ.get("ENABLE_AI_WARMUP", "True").lower() in ("true", "1", "yes")
        if os.environ.get("DISABLE_WARMUP", "0") == "1":
            enable_warmup = False

        if not enable_warmup:
            logger.info("AI Warmup DISABLED via environment variable.")
            return

        def warmup():
            # Döngüsel importu önlemek için içeride import
            try:
                # 2 saniye bekle ki Django tamamen yüklensin
                time.sleep(2)
                logger.info("AI Warmup Triggering...")
                from sinema_sitesi.ai_client import load_model
                load_model()
            except Exception as e:
                logger.warning(f"Model warmup hatası: {e}")

        # Daemon thread: Ana program kapanınca bu da kapansın
        t = threading.Thread(target=warmup, daemon=True)
        t.start()

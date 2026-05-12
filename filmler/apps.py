from django.apps import AppConfig
import threading
import logging
import time
import os
import sys

logger = logging.getLogger(__name__)


class FilmlerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "filmler"
    verbose_name = "Film Yönetimi"

    SKIP_COMMANDS = {
        'migrate', 'makemigrations', 'collectstatic',
        'test', 'check', 'shell', 'dbshell', 'flush',
        'showmigrations', 'sqlmigrate', 'film_cek',
    }

    def ready(self):
        # Yönetim komutlarında modeli yüklemeye çalışma
        if len(sys.argv) > 1 and sys.argv[1] in self.SKIP_COMMANDS:
            logger.info(f"AI Warmup SKIPPED (Command: {sys.argv[1]})")
            return

        # runserver autoreload'da ready() iki kez çağrılır;
        # sadece child process'te warmup yap (RUN_MAIN=true yalnızca ikinci süreçte set olur).
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            return

        # Env ile devre dışı bırakma
        if os.environ.get("DISABLE_WARMUP", "0") == "1":
            logger.info("AI Warmup DISABLED via environment variable.")
            return

        if os.environ.get("ENABLE_AI_WARMUP", "True").lower() not in ("true", "1", "yes"):
            logger.info("AI Warmup DISABLED via environment variable.")
            return

        def warmup():
            try:
                time.sleep(2)
                logger.info("AI Warmup Triggering...")
                from sinema_sitesi.ai_client import load_model
                load_model()
            except Exception as e:
                logger.warning(f"Model warmup hatasi: {e}")

        threading.Thread(target=warmup, daemon=True).start()

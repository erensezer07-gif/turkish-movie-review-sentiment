#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────
# Render Start Script — SezerFilm
# ──────────────────────────────────────────────────────────────────────
# Build aşamasında migrate çalıştırıldı ve şema image'e gömüldü.
# Yine de Render Free Tier'da disk Ephemeral; restart'ta DB sıfırlanır.
# Bu yüzden burada da idempotent migrate + film_cek + superuser çalıştırırız.
# ──────────────────────────────────────────────────────────────────────

echo "▶ Migrate (idempotent)"
python manage.py migrate --noinput

echo "▶ Süper kullanıcı garantisi (env vars varsa)"
python manage.py ensure_superuser || echo "  (atlandı)"

echo "▶ İlk filmler çekiliyor (TMDB sayfa 1)"
python manage.py film_cek 1 || echo "  (TMDB hatası, devam)"

echo "▶ Gunicorn başlatılıyor"
exec gunicorn sinema_sitesi.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --timeout 600 \
    --workers 1 \
    --threads 2 \
    --preload

#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Render Build Script — SezerFilm
# ──────────────────────────────────────────────────────────────────────
# 1) CPU-only torch (CUDA wheel'leri 2GB+ — Render Free Tier'a sığmaz)
# 2) Geri kalan bağımlılıklar
# 3) BERT modeli (build image'a gömülü kalır → her uyanışta hazır)
# 4) DB şeması + statik dosyalar
# ──────────────────────────────────────────────────────────────────────

echo "▶ pip + setuptools güncelleniyor"
python -m pip install --upgrade pip setuptools wheel

echo "▶ Torch (CPU) kuruluyor"
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu

echo "▶ Diğer bağımlılıklar kuruluyor"
python -m pip install -r requirements.txt

echo "▶ BERT modeli indiriliyor (Google Drive)"
python scripts/download_model.py

echo "▶ Veritabanı şeması uygulanıyor"
python manage.py migrate --noinput

echo "▶ Statik dosyalar toplanıyor"
python manage.py collectstatic --noinput

echo "✓ Build tamamlandı"

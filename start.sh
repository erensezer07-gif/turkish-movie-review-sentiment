#!/usr/bin/env bash
set -e

MODEL_DIR=/tmp/bert_model

if [ ! -d "$MODEL_DIR" ]; then
  echo "Downloading BERT model..."
  python -m pip install -q gdown==5.2.0
  gdown --fuzzy "https://drive.google.com/file/d/1r3Hxo_fnYWu0VtHZgpPne_7IaY2EQ21W/view?usp=sharing" -O /tmp/bert.zip
  echo "Unzipping model..."
  mkdir -p "$MODEL_DIR"
  unzip -q /tmp/bert.zip -d "$MODEL_DIR"
fi

export BERT_MODEL_PATH="$MODEL_DIR"
export TFIDF_PATH="/opt/render/project/src/yapay_zeka_servisi/film_tfidf_3cls.pkl"

python manage.py migrate --noinput
python manage.py collectstatic --noinput

gunicorn sinema_sitesi.wsgi:application --bind 0.0.0.0:$PORT --timeout 300

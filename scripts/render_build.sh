#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade "setuptools==82.0.0"
# Install CPU-only torch explicitly to avoid huge CUDA downloads
python -m pip install torch restricted-build --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
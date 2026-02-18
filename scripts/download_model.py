import os
import sys
import shutil
import gdown
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SERVICE_DIR = BASE_DIR / "yapay_zeka_servisi"
MODEL_DIR = SERVICE_DIR / "benim_bert_modelim_3cls_v2"
TEMP_ZIP = SERVICE_DIR / "bert_model.zip"

# Google Drive File ID (from render.yaml / start.sh)
FILE_ID = "1r3Hxo_fnYWu0VtHZgpPne_7IaY2EQ21W"
URL = f'https://drive.google.com/uc?id={FILE_ID}'

def main():
    print(f"Checking model directory: {MODEL_DIR}")
    
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print("Model already exists. Skipping download.")
        return

    print("Model not found. Downloading from Google Drive...")
    
    # Ensure service dir exists
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        gdown.download(URL, str(TEMP_ZIP), quiet=False, fuzzy=True)
        
        print("Unzipping model...")
        shutil.unpack_archive(str(TEMP_ZIP), str(SERVICE_DIR))
        
        # Cleanup
        os.remove(TEMP_ZIP)
        
        print(f"Model successfully downloaded to: {MODEL_DIR}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

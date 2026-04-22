# ml_workspace/scripts/export_model.py

import pickle
import os
import shutil

SRC = "backend/app/ml/artifacts"
DST = "exports"

def export_artifacts():
    os.makedirs(DST, exist_ok=True)
    for fname in ["model.pkl", "vectorizer.pkl", "label_encoder.pkl"]:
        shutil.copy(f"{SRC}/{fname}", f"{DST}/{fname}")
        print(f"Exported: {fname}")

if __name__ == "__main__":
    export_artifacts()
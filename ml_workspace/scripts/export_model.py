import shutil
import os

SRC_DIR = "."
DEST_DIR = "../artifacts"

files = [
    "model.pkl",
    "vectorizer.pkl",
    "label_encoder.pkl"
]

os.makedirs(DEST_DIR, exist_ok=True)

for file in files:
    src_path = os.path.join(SRC_DIR, file)
    dest_path = os.path.join(DEST_DIR, file)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"{file} not found. Train model first.")

    shutil.copy(src_path, dest_path)
    print(f"✅ {file} exported to artifacts")

print("🎉 All files exported successfully")

# ml_workspace/scripts/evaluate.py

import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ARTIFACTS_DIR = "backend/app/ml/artifacts"

def load_artifacts():
    with open(f"{ARTIFACTS_DIR}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

def evaluate(X_raw, y_true_labels):
    model, vectorizer, label_encoder = load_artifacts()
    X_vec = vectorizer.transform(X_raw)
    y_encoded = label_encoder.transform(y_true_labels)
    y_pred = model.predict(X_vec)
    acc = accuracy_score(y_encoded, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))
    return acc

if __name__ == "__main__":
    print("Run evaluate() with your test data arrays.")
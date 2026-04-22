import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "backend/app/ml/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─── TEXT PREPROCESSING ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def build_feature_text(row):
    category = clean_text(str(row["category"]))
    ingredients = clean_text(str(row["ingredients"]))
    return f"{category} {category} {category} {ingredients}"

# ─── LOAD CSV DATA ──────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}: {len(df)} rows")

    df["feature_text"] = df.apply(build_feature_text, axis=1)
    return df

# ─── TRAIN MODEL ────────────────────────────────────────────────────────
def train(csv_path: str):
    df = load_data(csv_path)

    X = df["feature_text"].values
    y = df["dish"].values

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )

    X_vec = vectorizer.fit_transform(X)

    # Train/Test Split
    class_counts = Counter(y_encoded)
    use_stratify = min(class_counts.values()) >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded if use_stratify else None
    )

    # Models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)

    model = LogisticRegression(max_iter=1000)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # Cross Validation
    if min(class_counts.values()) >= 2:
        cv_scores = cross_val_score(
            model,
            X_vec,
            y_encoded,
            cv=min(5, min(class_counts.values())),
            scoring="accuracy"
        )
        print(f"Cross-val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    else:
        print("⚠️ Cross-validation skipped: each dish needs at least 2 samples")

    # Save Artifacts
    with open(f"{ARTIFACTS_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{ARTIFACTS_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"{ARTIFACTS_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\n✅ Artifacts saved to {ARTIFACTS_DIR}")

# ─── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train("ml_workspace/data/dishes.csv")
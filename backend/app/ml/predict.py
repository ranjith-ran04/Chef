# backend/app/ml/predict.py

import pickle
import os
import re
import numpy as np
from typing import List, Dict, Any

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ─── LOAD ARTIFACTS (cached at module level) ──────────────────────────────────
def _load():
    with open(f"{ARTIFACTS_DIR}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

try:
    _MODEL, _VECTORIZER, _LABEL_ENCODER = _load()
except Exception as e:
    _MODEL = _VECTORIZER = _LABEL_ENCODER = None
    print(f"[WARN] Could not load ML artifacts: {e}")

# ─── TEXT UTILS ───────────────────────────────────────────────────────────────
STOPWORDS = {"a", "an", "the", "and", "or", "with", "of", "in", "on", "some", "fresh"}

def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def _normalize_ingredients(ingredients: List[str]) -> str:
    parts = []
    for ing in ingredients:
        tokens = [t for t in _clean(ing).split() if t not in STOPWORDS]
        parts.extend(tokens)
    return " ".join(parts)

def _build_feature(category: str, ingredients: List[str]) -> str:
    cat = _clean(category)
    ings = _normalize_ingredients(ingredients)
    return f"{cat} {cat} {cat} {ings}"

# ─── DIVERSITY FILTER ─────────────────────────────────────────────────────────
def _are_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Simple word-overlap similarity to filter near-duplicate dish names."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap >= threshold

def _diverse_top_k(probs: np.ndarray, classes: np.ndarray, k: int = 3) -> List[Dict]:
    sorted_idx = np.argsort(probs)[::-1]
    selected = []
    for idx in sorted_idx:
        dish = classes[idx]
        conf = round(float(probs[idx]) * 100, 1)
        if conf < 1.0:
            break
        # Skip if too similar to already-selected dish
        if any(_are_similar(dish, s["dish"]) for s in selected):
            continue
        selected.append({"dish": dish, "confidence": conf})
        if len(selected) == k:
            break
    return selected

# ─── EXPLANATION GENERATOR ────────────────────────────────────────────────────
def _generate_explanation(dish: str, category: str, ingredients: List[str], confidence: float) -> str:
    top_ings = ingredients[:3]
    ing_str = ", ".join(top_ings)
    conf_label = (
        "strongly" if confidence >= 85
        else "confidently" if confidence >= 70
        else "moderately" if confidence >= 50
        else "tentatively"
    )
    return (
        f"Based on the {category} category and key ingredients like {ing_str}, "
        f"the model {conf_label} predicts '{dish}' as the best match "
        f"with a confidence of {confidence:.1f}%."
    )

# ─── MAIN PREDICT FUNCTION ────────────────────────────────────────────────────
def predict_dish(category: str, ingredients: List[str]) -> Dict[str, Any]:
    if _MODEL is None:
        raise RuntimeError("ML model artifacts not loaded. Please train the model first.")

    feature_text = _build_feature(category, ingredients)
    X_vec = _VECTORIZER.transform([feature_text])

    probs = _MODEL.predict_proba(X_vec)[0]
    classes = _LABEL_ENCODER.classes_

    recommendations = _diverse_top_k(probs, classes, k=3)

    if not recommendations:
        raise ValueError("No predictions could be generated.")

    best = recommendations[0]
    explanation = _generate_explanation(
        dish=best["dish"],
        category=category,
        ingredients=ingredients,
        confidence=best["confidence"],
    )

    return {
        "best_match": best["dish"],
        "recommendations": recommendations,
        "explanation": explanation,
    }
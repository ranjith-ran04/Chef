# backend/app/ml/predict.py

import pickle
import os
import re
import numpy as np # type: ignore
from typing import List, Dict, Any

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ─── RECIPE DATABASE ──────────────────────────────────────────────────────────
RECIPE_DB: Dict[str, Dict] = {
    "paneer butter masala": {
        "ingredients": ["paneer", "tomato puree", "butter", "cream", "onion", "garlic", "ginger", "garam masala", "kasuri methi", "salt"],
        "steps": [
            "Melt butter in a pan over medium heat.",
            "Sauté finely chopped onions until golden brown.",
            "Add ginger-garlic paste and cook for 2 minutes.",
            "Pour in tomato puree and cook until oil separates.",
            "Add garam masala, red chili powder, and salt.",
            "Stir in fresh cream and mix well.",
            "Add cubed paneer and simmer for 5 minutes.",
            "Finish with crushed kasuri methi. Serve hot with naan or rice.",
        ],
    },
    "kadai paneer": {
        "ingredients": ["paneer", "capsicum", "onion", "tomato", "kadai masala", "ginger", "garlic", "oil", "coriander", "salt"],
        "steps": [
            "Heat oil in a kadai (wok) over high heat.",
            "Add diced onions and sauté until translucent.",
            "Add ginger-garlic paste and cook for 1 minute.",
            "Add chopped tomatoes and cook until mushy.",
            "Sprinkle kadai masala and stir well.",
            "Add diced capsicum and cook for 3 minutes keeping it slightly crunchy.",
            "Add paneer cubes and toss gently to coat.",
            "Garnish with fresh coriander and serve.",
        ],
    },
    "shahi paneer": {
        "ingredients": ["paneer", "cream", "cashew paste", "onion", "tomato", "saffron", "cardamom", "butter", "sugar", "salt"],
        "steps": [
            "Soak saffron in warm milk and set aside.",
            "Heat butter and sauté onions until soft.",
            "Add cashew paste and cook for 3 minutes.",
            "Blend in tomato puree and simmer for 5 minutes.",
            "Add cardamom powder, sugar, and salt.",
            "Pour in saffron milk and cream, stir gently.",
            "Add paneer cubes and cook on low heat for 4 minutes.",
            "Serve garnished with silver leaf or saffron strands.",
        ],
    },
    "butter chicken": {
        "ingredients": ["chicken", "tomato", "butter", "cream", "garlic", "ginger", "garam masala", "kasuri methi", "yogurt", "salt"],
        "steps": [
            "Marinate chicken in yogurt, ginger-garlic paste and spices for 2 hours.",
            "Grill or pan-fry chicken until charred slightly. Set aside.",
            "Melt butter in a pan and sauté garlic.",
            "Add tomato puree and cook for 10 minutes until thick.",
            "Blend the sauce smooth, return to pan.",
            "Add cream, garam masala, and kasuri methi.",
            "Add the cooked chicken pieces and simmer for 8 minutes.",
            "Finish with a dollop of butter. Serve with naan.",
        ],
    },
    "dal tadka": {
        "ingredients": ["lentils", "tomato", "onion", "garlic", "cumin seeds", "turmeric", "ghee", "red chili", "coriander", "salt"],
        "steps": [
            "Rinse and pressure cook lentils with turmeric and salt until soft.",
            "Mash lentils lightly and keep on low heat.",
            "Heat ghee in a small pan for the tadka.",
            "Add cumin seeds and let them splutter.",
            "Add sliced garlic and dry red chilies, fry until golden.",
            "Add chopped onions and cook until brown.",
            "Add tomatoes and cook until mushy.",
            "Pour the tadka over the dal. Garnish with coriander.",
        ],
    },
    "pasta marinara": {
        "ingredients": ["pasta", "tomato", "garlic", "basil", "olive oil", "onion", "oregano", "salt", "pepper", "parmesan"],
        "steps": [
            "Boil pasta in salted water until al dente. Reserve 1 cup pasta water.",
            "Heat olive oil in a pan over medium heat.",
            "Sauté minced garlic for 1 minute until fragrant.",
            "Add crushed tomatoes, oregano, salt, and pepper.",
            "Simmer sauce for 15 minutes, stirring occasionally.",
            "Toss cooked pasta into the sauce with a splash of pasta water.",
            "Tear fresh basil leaves over the top.",
            "Serve with grated parmesan.",
        ],
    },
    "carbonara": {
        "ingredients": ["pasta", "eggs", "bacon", "parmesan", "black pepper", "garlic", "olive oil", "salt", "pecorino"],
        "steps": [
            "Cook pasta in salted boiling water until al dente.",
            "Whisk eggs with grated parmesan and pecorino. Season with black pepper.",
            "Fry diced bacon (guanciale) in olive oil until crispy.",
            "Remove pan from heat. Add drained pasta to bacon.",
            "Quickly pour egg mixture over pasta, tossing rapidly.",
            "Add pasta water little by little to create a creamy sauce.",
            "Do not reheat — the residual heat cooks the eggs.",
            "Serve immediately with extra parmesan and black pepper.",
        ],
    },
    "vegetable chow mein": {
        "ingredients": ["noodles", "mixed vegetables", "soy sauce", "garlic", "ginger", "sesame oil", "oyster sauce", "spring onion", "oil", "pepper"],
        "steps": [
            "Boil noodles until just cooked. Rinse under cold water and toss with oil.",
            "Heat oil in a wok over high heat until smoking.",
            "Stir-fry garlic and ginger for 30 seconds.",
            "Add vegetables and toss on high heat for 2 minutes.",
            "Push vegetables aside, add noodles to the wok.",
            "Pour soy sauce and oyster sauce over noodles.",
            "Toss everything together vigorously for 2 minutes.",
            "Drizzle sesame oil, garnish with spring onions. Serve hot.",
        ],
    },
    "egg fried rice": {
        "ingredients": ["rice", "eggs", "mixed vegetables", "soy sauce", "garlic", "sesame oil", "spring onion", "oil", "white pepper", "salt"],
        "steps": [
            "Use day-old cold cooked rice for best results.",
            "Beat eggs with a pinch of salt.",
            "Heat oil in a wok over high heat.",
            "Scramble eggs quickly and break into small pieces. Set aside.",
            "Add more oil, fry garlic until golden.",
            "Add vegetables and stir-fry for 2 minutes.",
            "Add rice, breaking any clumps, and toss on high heat.",
            "Add soy sauce, white pepper, and return eggs. Toss well. Finish with sesame oil.",
        ],
    },
    "chicken tacos": {
        "ingredients": ["tortillas", "chicken", "salsa", "avocado", "cheese", "lime", "cumin", "garlic", "oil", "coriander"],
        "steps": [
            "Season chicken with cumin, garlic powder, salt, and pepper.",
            "Grill or pan-fry chicken until cooked through. Rest for 5 minutes.",
            "Slice chicken into thin strips.",
            "Warm tortillas on a dry skillet for 30 seconds each side.",
            "Mash avocado with lime juice and salt.",
            "Assemble tacos: spread avocado, add chicken strips.",
            "Top with salsa and shredded cheese.",
            "Garnish with fresh coriander and a squeeze of lime.",
        ],
    },
    "thai green curry": {
        "ingredients": ["chicken", "coconut milk", "green curry paste", "lemongrass", "galangal", "kaffir lime leaves", "fish sauce", "basil", "oil", "palm sugar"],
        "steps": [
            "Heat oil in a wok and fry green curry paste for 2 minutes.",
            "Add sliced chicken and stir-fry until sealed.",
            "Pour in coconut milk and bring to a gentle simmer.",
            "Add bruised lemongrass, galangal slices, and kaffir lime leaves.",
            "Season with fish sauce and palm sugar.",
            "Simmer for 15 minutes until chicken is cooked through.",
            "Add Thai basil leaves and stir.",
            "Serve with steamed jasmine rice.",
        ],
    },
}

# ─── FALLBACK: Generic recipe builder ─────────────────────────────────────────
def _build_fallback_recipe(dish: str, ingredients: List[str]) -> Dict:
    return {
        "ingredients": ingredients,
        "steps": [
            f"Gather and prepare all ingredients: {', '.join(ingredients[:5])}.",
            "Heat oil or butter in a suitable pan over medium heat.",
            "Sauté aromatics (onion, garlic, ginger) until fragrant.",
            "Add main ingredients and cook according to their texture.",
            f"Season to taste and cook until {dish} is fully prepared.",
            "Garnish as desired and serve hot.",
        ],
    }

def _get_recipe(dish: str, ingredients: List[str]) -> Dict:
    key = dish.lower().strip()
    if key in RECIPE_DB:
        return RECIPE_DB[key]
    # Find closest recipe by word overlap
    best_match_key = None
    best_score = 0
    dish_words = set(key.split())
    for db_key in RECIPE_DB:
        overlap = len(dish_words & set(db_key.split()))
        if overlap > best_score:
            best_score = overlap
            best_match_key = db_key
    if best_match_key and best_score > 0:
        return RECIPE_DB[best_match_key]
    return _build_fallback_recipe(dish, ingredients)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
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
    recipe = _get_recipe(best["dish"], ingredients)

    return {
        "best_match": best["dish"],
        "recommendations": recommendations,
        "explanation": explanation,
        "recipe": recipe,
    }
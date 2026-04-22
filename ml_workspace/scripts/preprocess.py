# ml_workspace/scripts/preprocess.py

import re
import pandas as pd

STOPWORDS = {"a", "an", "the", "and", "or", "with", "of", "in", "on", "some"}

def normalize_ingredient(ingredient: str) -> str:
    ingredient = ingredient.lower().strip()
    ingredient = re.sub(r"[^a-z0-9\s]", "", ingredient)
    ingredient = re.sub(r"\s+", " ", ingredient)
    tokens = [w for w in ingredient.split() if w not in STOPWORDS]
    return " ".join(tokens)

def normalize_ingredients_list(ingredients) -> str:
    if isinstance(ingredients, list):
        return " ".join(normalize_ingredient(i) for i in ingredients)
    return normalize_ingredient(str(ingredients))

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["category", "ingredients", "dish"])
    df["category"] = df["category"].str.lower().str.strip()
    df["dish"] = df["dish"].str.lower().str.strip()
    df["ingredients"] = df["ingredients"].apply(normalize_ingredients_list)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df
import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z, ]', '', text)
    return text

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    df['ingredients'] = df['ingredients'].apply(clean_text)
    df['dish'] = df['dish'].str.lower()

    return df

if __name__ == "__main__":
    df = load_and_preprocess("../data/dishes.csv")
    print(df.head())





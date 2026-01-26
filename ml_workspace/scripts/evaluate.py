import joblib
import re

# Load artifacts
model = joblib.load("../artifacts/model.pkl")
vectorizer = joblib.load("../artifacts/vectorizer.pkl")
label_encoder = joblib.load("../artifacts/label_encoder.pkl")

# Simple preprocessing function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z, ]', '', text)
    return text

# Interactive loop
print("🍽️ Dish Prediction Model")
print("Type 'exit' to quit\n")

while True:
    category = input("Enter category: ")
    if category.lower() == "exit":
        break

    ingredients = input("Enter ingredients (comma separated): ")
    if ingredients.lower() == "exit":
        break

    # Preprocess input
    category = category.lower().strip()
    ingredients = clean_text(ingredients)
    input_text = f"category {category} ingredients {ingredients}"

    # Vectorize & predict
    input_vec = vectorizer.transform([input_text])
    pred = model.predict(input_vec)
    dish_name = label_encoder.inverse_transform(pred)[0]

    print(f"✅ Predicted Dish: {dish_name}\n")

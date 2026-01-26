import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess

# Load data
df = load_and_preprocess("../data/dishes.csv")

# Combine category + ingredients
df["input_text"] = (
    "category " + df["category"] +
    " ingredients " + df["ingredients"]
)

X = df["input_text"]
y = df["dish"]

# Encode dish labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ❌ NO stratify (dataset small)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save files
joblib.dump(model, "./artifactsmodel.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Training completed successfully")

import joblib
from train_model import model

def export():
    joblib.dump(model, "../artifacts/model.pkl")
    print("Model exported successfully")

if __name__ == "__main__":
    export()

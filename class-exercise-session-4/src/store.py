import joblib
from datetime import datetime
from src.metadata import MODELS_FOLDER

def store_model(model, model_name: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"{MODELS_FOLDER}/class_model-{model_name}-{timestamp}.joblib"
    joblib.dump(model, model_path)
    print(f"Model stored as: {model_path}")

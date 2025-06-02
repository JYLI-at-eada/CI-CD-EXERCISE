import pandas as pd
from metadata import DATASETS_FOLDER

def load_data(file_name: str) -> pd.DataFrame:
    file_path = f"{DATASETS_FOLDER}/{file_name}"
    df = pd.read_csv(file_path)
    print(f"Data loaded from: {file_path} — shape: {df.shape}")
    return df
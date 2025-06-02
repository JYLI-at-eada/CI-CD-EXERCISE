from src.load import load_data
from src.transform import Transformer, balance_dataset
from src.train import train_model
from src.store import store_model
from src.metadata import MODEL_NAME


def main() -> None:
    df = load_data("Churn_Modelling_train_test.csv")
    df = Transformer().transform(df)   # 生成并保留 y
    df = balance_dataset(df)           # 按 y 平衡
    model = train_model(df, target_column="y")
    store_model(model, MODEL_NAME)


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.metadata import (
    COLUMNS_TO_DROP,
    BINARY_FEATURES,
    ONE_HOT_ENCODE_COLUMNS,
)

_POSSIBLE_LABELS = ["y", "Exited", "Churn", "deposit", "target"]


class Transformer:
    def __init__(self) -> None:
        self.drop_columns = COLUMNS_TO_DROP
        self.binary_variable_columns = BINARY_FEATURES
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS

    # ----------------------------- pipeline ----------------------------- #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "y" not in df.columns:
            label = next((c for c in _POSSIBLE_LABELS if c in df.columns), None)
            if label is None:
                raise KeyError(f"缺少标签列（候选: {_POSSIBLE_LABELS}）")
            if set(df[label].unique()) <= {0, 1}:
                df["y"] = df[label].map({1: "yes", 0: "no"})
            else:
                df["y"] = df[label]

        y_backup = df["y"].copy()
        df = df.drop(self.drop_columns, axis=1, errors="ignore")
        df = self._map_binary_column_to_int(df)
        df = self._map_month_to_int(df)
        df = self._one_hot_encoding(df)
        df["y"] = y_backup
        return df

    # ---------------------------- helpers ------------------------------- #
    def _map_binary_column_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.binary_variable_columns:
            if col in df.columns:
                df[col] = df[col].map({"yes": 1, "no": 0})
        return df

    def _map_month_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        if "month" in df.columns:
            df["month"] = df["month"].map(
                {"jan": 1, "feb": 2, "mar": 3, "apr": 4,
                 "may": 5, "jun": 6, "jul": 7, "aug": 8,
                 "sep": 9, "oct": 10, "nov": 11, "dec": 12}
            )
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.one_hot_encoding_columns if c in df.columns]
        if cols:
            enc = (
                OneHotEncoder(drop="first", sparse_output=False)
                .set_output(transform="pandas")
            )
            enc.fit(df[cols])
            df = pd.concat([df.drop(columns=cols), enc.transform(df[cols])], axis=1)

     
        obj_cols = df.select_dtypes("object").columns.tolist()
        if obj_cols:
            df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
        return df


# ------------------------- balance function ---------------------------- #
def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_y0 = df[df["y"] == "no"].copy()
    df_y1 = df[df["y"] == "yes"].copy()
    if df_y0.empty or df_y1.empty:
        return df
    n = len(df_y1)
    df_bal = pd.concat(
        [df_y0.sample(n=n, random_state=42), df_y1]
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_bal

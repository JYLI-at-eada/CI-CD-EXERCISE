MODELS_FOLDER = "models"
DATASETS_FOLDER = "datasets"
MODEL_NAME = "seline-logistic"

COLUMNS_TO_DROP = []
BINARY_FEATURES = [
    "housing",
    "loan",
    "default",
]
ONE_HOT_ENCODE_COLUMNS = [
    "marital",
    "job",
    "education",
    "poutcome",
    "contact",
]
MODEL_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

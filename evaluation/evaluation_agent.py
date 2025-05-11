import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import joblib
import os

def load_dataset(path="data/adult.csv", target_col="income"):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(path)

    df = df.dropna()

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_evaluation_agent(X_train, X_test, y_train, y_test, config: dict):
    """Trains and evaluates the selected model."""
    algo = config["algorithm"]
    params = config["params"]

    if algo == "RandomForest":
        model = RandomForestClassifier(**params)
    elif algo == "XGBoost":
        model = XGBRegressor(**params)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {}

    if algo == "RandomForest":
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["report"] = classification_report(y_test, y_pred)

    os.makedirs("output", exist_ok=True)
    joblib.dump(model, "output/model.pkl")

    return results

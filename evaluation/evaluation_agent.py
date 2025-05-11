import time
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_evaluation_agent(X_train, X_test, y_train, y_test, model_config, output_dir="output", dataset_name="dataset"):
    """
    Trains and evaluates a model using provided config and data.
    Saves model and returns results as a dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    algo = model_config["algorithm"]
    params = model_config["params"]

    # Choose the model
    if algo == "RandomForest":
        model = RandomForestClassifier(**params)
    elif algo == "LogisticRegression":
        model = LogisticRegression(**params)
    elif algo == "XGBoost":
        model = XGBClassifier(**params, eval_metric='logloss')
    elif algo == "LightGBM":
        model = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Train and time
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = round(end - start, 4)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    model_path = os.path.join(output_dir, f"{dataset_name}_{algo}.pkl")
    joblib.dump(model, model_path)

    results = {
        "dataset": dataset_name,
        "algorithm": algo,
        "accuracy": acc,
        "train_time_sec": train_time,
        "model_path": model_path,
        "report": report
    }

    return results

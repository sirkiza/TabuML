import os
import time
import joblib
from typing import Optional, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def _choose_model(algo: str, params: dict):
    if algo == "RandomForest":
        return RandomForestClassifier(**params)
    if algo == "LogisticRegression":
        return LogisticRegression(**params)
    if algo == "XGBoost":
        return XGBClassifier(**params, eval_metric="logloss")
    if algo == "LightGBM":
        return LGBMClassifier(**params)
    raise ValueError(f"Unsupported algorithm: {algo}")

def run_evaluation_agent(
    X_train,
    X_test,
    y_train,
    y_test,
    model_config: dict,
    output_dir: str = "output",
    dataset_name: str = "dataset",
):
    os.makedirs(output_dir, exist_ok=True)

    algo = model_config["algorithm"]
    params = model_config["params"]
    model = _choose_model(algo, params)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 4)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    y_proba: Optional[List[List[float]]] = None
    proba_labels: Optional[List] = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test).tolist()
        proba_labels = model.classes_.tolist()

    model_path = os.path.join(output_dir, f"{dataset_name}_{algo}.pkl")
    joblib.dump(model, model_path)

    results = {
        "dataset": dataset_name,
        "algorithm": algo,
        "accuracy": acc,
        "train_time_sec": train_time,
        "model_path": model_path,
        "predictions": y_pred.tolist(),
        "probabilities": y_proba,
        "probabilities_labels": proba_labels,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "model": model
    }

    return results

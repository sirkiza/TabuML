import pandas as pd
from evaluation.dataset_loader import load_dataset
from agents.tabuml_automl import TabuMLAutoML

def run_model_agent(dataset_path: str, target_column: str, dataset_name: str):
    # Load and preprocess dataset
    X, y, _ = load_dataset(dataset_path, target_column)
    df = pd.DataFrame(X).copy()
    df[target_column] = y

    # Fit LLM-guided AutoML
    automl = TabuMLAutoML(dataset_name=dataset_name)
    automl.fit(df.drop(columns=[target_column]), df[target_column])

    # Predict on the same dataset
    predictions, probabilities = automl.predict(df.drop(columns=[target_column]))

    # Return predictions and metadata
    return {
        **automl.model_output,
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist() if probabilities is not None else None,
        "truth": y.tolist()
    }

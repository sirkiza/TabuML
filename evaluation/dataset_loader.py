import pandas as pd
import os

def load_dataset(path, target_column):
    """
    Loads a CSV file into features and target.
    Returns: X, y, metadata_dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset.")

    # Drop rows with any missing values (simplest for now)
    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    metadata = {
        "path": path,
        "target_column": target_column,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "class_distribution": y.value_counts(normalize=True).to_dict() if y.nunique() <= 20 else "continuous",
        "num_categorical": len(X.select_dtypes(include='object').columns),
        "num_numerical": len(X.select_dtypes(include='number').columns),
        "missing_values_before_drop": df.isnull().sum().sum()
    }

    return X, y, metadata

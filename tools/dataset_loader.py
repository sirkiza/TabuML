import pandas as pd
import os
from scipy.io import arff

def load_dataset(path, target_column):
    if path.endswith('.arff'):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        # Decode byte strings (common in ARFF)
        for col in df.select_dtypes([object]):
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        df = pd.read_csv(path)
        y = df[target_column]
        X = df.drop(columns=[target_column])
        meta = {
            "path": path,
            "target_column": target_column,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "class_distribution": y.value_counts(normalize=True).to_dict() if y.nunique() <= 20 else "continuous",
            "num_categorical": len(X.select_dtypes(include='object').columns),
            "num_numerical": len(X.select_dtypes(include='number').columns),
            "missing_values_before_drop": df.isnull().sum().sum()
        }

    return X, y, meta

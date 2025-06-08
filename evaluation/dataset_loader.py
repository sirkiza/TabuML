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
    else:
        df = pd.read_csv(path)
        meta = None

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y, meta

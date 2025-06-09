import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(X: pd.DataFrame, y, model_type: str):
    """
    Preprocesses X and y based on model type.
    Returns: transformed X, transformed y, and the fitted preprocessing pipeline.
    """
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Define imputers
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="mean")

    if model_type in ["RandomForest", "DecisionTree"]:
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_imputer, numerical_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_cols)
        ])
    elif model_type in ["LogisticRegression", "MLP", "SVM"]:
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', num_imputer),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ])
    elif model_type in ["XGBoost", "LightGBM"]:
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_imputer, numerical_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_cols)
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Check if we have at least one valid transformer
    if not numerical_cols and not categorical_cols:
        print(f"[Preprocessing Debug] No usable columns found in X:\n{X.head()}")
        print(f"[Preprocessing Debug] Dtypes:\n{X.dtypes}")
        raise ValueError("No valid feature transformers could be created. Check if input DataFrame has usable columns.")

    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)

    # Encode target
    if y.dtype.kind in 'O' or y.dtype.name == 'category':
        y = pd.Series(LabelEncoder().fit_transform(y))
    else:
        y = pd.Series(y).astype(float if y.dtype.kind in 'f' else int)

    return X_processed, y, preprocessor

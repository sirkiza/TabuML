import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(X: pd.DataFrame, y, model_type: str):
    """
    Preprocesses X based on model type.
    Returns: transformed X and fitted pipeline (for later reuse or export)
    """
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Default strategies
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="mean")

    if model_type in ["RandomForest", "DecisionTree"]:
        # Trees are insensitive to scaling and can handle ordinal encoding
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_imputer, numerical_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_cols)
        ])

    elif model_type in ["LogisticRegression", "MLP", "SVM"]:
        # These benefit from scaling + one-hot encoding
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', num_imputer),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), categorical_cols)
        ])

    elif model_type in ["XGBoost", "LightGBM"]:
        # Typically prefer numeric input; ordinal encoding works
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_imputer, numerical_cols),
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_cols)
        ])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

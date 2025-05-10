import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_dataset(path='data/adult.csv', target_col='income'):
    df = pd.read_csv(path)
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_evaluation_agent(X_train, X_test, y_train, y_test, config: dict):
    if config['algorithm'] == 'RandomForest':
        model = RandomForestClassifier(**config['params'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        joblib.dump(model, 'output/model.pkl')
        return {'accuracy': acc, 'report': report}

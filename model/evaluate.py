import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from model.train import preprocess_data, prepare_features
import os

# Chemins absolus
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "data", "processed", "nyc_taxi.db")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ridge_model.joblib")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model, features = pickle.load(file)  # Décompressez le tuple
    print(f"Done")
    return model

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data_test = pd.read_sql('SELECT * FROM test', con)
    con.close()
    X = data_test.drop(columns=['target'])
    y = data_test['target']
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])

    return X, y

def evaluate_model(model, X, y):
    print(f"Evaluating the model")
    X_preprocessed = preprocess_data(X)
    X_test = prepare_features(X_preprocessed)  # Appliquez les mêmes transformations
    y_pred = model.predict(X_test)
    score = mean_squared_error(y, y_pred)
    return score

if __name__ == "__main__":

    X_test, y_test = load_test_data(DB_PATH)
    model = load_model(MODEL_PATH)
    score_test = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {score_test:.2f}")

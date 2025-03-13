import sqlite3
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import os, pickle
import yaml
from TaxiModel import TaxiModel

# Chemins absolus
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = os.path.join(ROOT_DIR, CONFIG['paths']['data'])
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ridge_model_custom.joblib")
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "New_York_City_Taxi_Trip_Duration.zip")

def load_train_data(path):
    print(f"Lecture des données d'entraînement depuis : {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    
    # Convertir pickup_datetime en datetime
    data_train['pickup_datetime'] = pd.to_datetime(data_train['pickup_datetime'])
    
    X = data_train.drop(columns=['target'])
    y = data_train['target']
    return X, y

def train_model(X, y):
    print(f"Construction du modèle...")
    model = TaxiModel(Ridge())
    model.fit(X, y)
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    print(f"Score sur les données d'entraînement : {score:.4f}")
    return model 

def persist_model(model, path):
    print(f"Sauvegarde du modèle dans {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Terminé")

if __name__ == "__main__":
    X_train, y_train = load_train_data(DB_PATH)
    model = train_model(X_train, y_train)
    persist_model(model, MODEL_PATH)
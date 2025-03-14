import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from datetime import datetime
import pandas as pd
import sys
import mlflow
import yaml
import sqlite3
import numpy as np

# Ajouter les répertoires pertinents au chemin de recherche Python
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
taxi_class_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'taxi_class'))
sys.path.insert(0, taxi_class_dir)

# Configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Configuration MLflow
MLFLOW_TRACKING_URI = "file:" + os.path.join(ROOT_DIR, CONFIG['paths']['mlruns'])
MODEL_NAME = CONFIG['mlflow']['model_name']

# Initialiser le client MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = mlflow.MlflowClient()

# Charger le dernier modèle du registre MLflow
model_metadata = mlflow_client.get_latest_versions(MODEL_NAME, stages=["None"])
latest_model_version = model_metadata[0].version
model_uri = f"models:/{MODEL_NAME}/{latest_model_version}"
model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Charger le modèle personnalisé
MODEL_PATH_CUSTOM = os.path.join(ROOT_DIR, "models", "ridge_model_custom.joblib")
with open(MODEL_PATH_CUSTOM, 'rb') as file:
    model_custom = pickle.load(file)  

PREDICTIONS_DB = os.path.join(ROOT_DIR, CONFIG['paths']['data'])

class InputModel(BaseModel):
    pickup_datetime: datetime


app = FastAPI()


@app.post("/predict")
async def predict(trip: InputModel):
    # Créer le DataFrame avec les données d'entrée
    temp_df = pd.DataFrame([{
        'pickup_datetime': trip.pickup_datetime
    }])
    
    # Préparation des features
    temp_df['weekday'] = temp_df['pickup_datetime'].dt.weekday
    temp_df['month'] = temp_df['pickup_datetime'].dt.month
    temp_df['hour'] = temp_df['pickup_datetime'].dt.hour
    
    # Identification des périodes anormales
    temp_df['abnormal_period'] = 0  # Par défaut, on suppose que ce n'est pas une période anormale
    
    # Sélection des features pour la prédiction
    input_data = temp_df[['abnormal_period', 'hour', 'weekday', 'month']]
    
    # Prédiction avec le modèle MLflow
    prediction_log = model.predict(input_data)[0]
    duree_secondes = np.expm1(prediction_log)  # Inverse de log1p
    duree_minutes = round(duree_secondes / 60, 1)
    heure_arrivee = trip.pickup_datetime + pd.Timedelta(minutes=duree_minutes)
    
    # Stockage dans SQLite
    conn = sqlite3.connect(PREDICTIONS_DB)
    c = conn.cursor()
    
    # Créer la table si elle n'existe pas
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  pickup_datetime TEXT,
                  predicted_duration REAL,
                  prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  model_version TEXT)''')
    
    # Insérer les données
    c.execute('''INSERT INTO predictions 
                 (pickup_datetime, predicted_duration, model_version)
                 VALUES (?, ?, ?)''',
              (trip.pickup_datetime.isoformat(), duree_minutes, f"MLflow_v{latest_model_version}"))
    print(f"Données insérées dans la base de données : {trip.pickup_datetime.isoformat()} - {duree_minutes}")
    
    conn.commit()
    conn.close()
    
    return {
        "duree_trajet_minutes": duree_minutes,
        "heure_arrivee_approximative": heure_arrivee,
        "version_modele": f"MLflow_v{latest_model_version}"
    }


@app.post("/predict_custom")
async def predict_custom(trip: InputModel):
    # Créer le DataFrame avec les données d'entrée
    temp_df = pd.DataFrame([{
        'pickup_datetime': trip.pickup_datetime
    }])
    
    duration_secondes = model_custom.predict(temp_df)[0]
         
    # Stockage dans SQLite
    conn = sqlite3.connect(PREDICTIONS_DB)
    c = conn.cursor()
    
    # Créer la table si elle n'existe pas
    c.execute('''CREATE TABLE IF NOT EXISTS predictions_custom
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  pickup_datetime TEXT,
                  predicted_duration REAL,
                  prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Insérer les données
    c.execute('''INSERT INTO predictions_custom 
                 (pickup_datetime, predicted_duration)
                 VALUES (?, ?)''',
              (trip.pickup_datetime.isoformat(), duration_secondes))
    print(f"Données insérées dans la base de données : {trip.pickup_datetime.isoformat()} - {duration_secondes}")
    
    conn.commit()
    conn.close()
    
    return {
        "duree_trajet_secon": duration_secondes,
        "modele": "custom"
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
    # Note: Pour lancer depuis la ligne de commande, utilisez:
    # cd src/api
    # uvicorn main:app --reload
    # 
    # OU depuis la racine du projet:
    # uvicorn src.api.main:app --reload
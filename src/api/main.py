import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from datetime import datetime
import pandas as pd
import sys

# Ajouter les répertoires pertinents au chemin de recherche Python
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
taxi_class_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'taxi_class'))
sys.path.insert(0, taxi_class_dir)

# Maintenant on peut importer depuis les modules requis
from train import prepare_features, inverse_transform_target
import yaml
import sqlite3

# Configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Charger le modèle
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ridge_model.joblib")
with open(MODEL_PATH, 'rb') as file: 
    model, features = pickle.load(file)

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
    temp_df = pd.DataFrame([{
        'pickup_datetime': trip.pickup_datetime
    }])
    
    processed = prepare_features(temp_df)
    input_data = processed[['abnormal_period', 'hour', 'weekday', 'month']]
    
    prediction_log = model.predict(input_data)[0]
    duree_secondes = inverse_transform_target(prediction_log)
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
                  prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Insérer les données
    c.execute('''INSERT INTO predictions 
                 (pickup_datetime, predicted_duration)
                 VALUES (?, ?)''',
              (trip.pickup_datetime.isoformat(), duree_minutes))
    print(f"Données insérées dans la base de données : {trip.pickup_datetime.isoformat()} - {duree_minutes}")
    
    conn.commit()
    conn.close()
    
    return {
        "duree_trajet_minutes": duree_minutes,
        "heure_arrivee approximative": heure_arrivee
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
    # Pour exécuter l'API directement à partir de ce fichier
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
    # Note: Pour lancer depuis la ligne de commande, utilisez:
    # cd src/api
    # uvicorn main:app --reload
    # 
    # OU depuis la racine du projet:
    # uvicorn src.api.main:app --reload
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from datetime import datetime
import pandas as pd
from model.train import prepare_features, inverse_transform_target
import numpy as np
import sqlite3

# Charger le modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ridge_model.joblib')
with open(MODEL_PATH, 'rb') as file: 
    model, features = pickle.load(file)

PREDICTIONS_DB = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'nyc_taxi.db')

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



if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
"""
Script pour entraîner le modèle
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.load_data import load_data, prepare_data
from src.preprocessing.preprocess import (
    extract_datetime_features, 
    transform_target
)
from src.models.train import train_model, evaluate_model, save_model
from config.model_config import FEATURES, SPLIT_PARAMS

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Préparer les features pour l'entraînement
    
    Args:
        df: DataFrame avec les données brutes
        
    Returns:
        DataFrame avec les features préparées
    """
    # Copier pour éviter de modifier l'original
    data = df.copy()
    
    # Extraire les features de date/heure
    data = extract_datetime_features(data)
    
    return data

def main():
    # Charger les données
    print("Chargement des données...")
    data = load_data("data/raw/New_York_City_Taxi_Trip_Duration.zip")
    data = prepare_data(data)
    
    # Préparer les features
    print("\nPréparation des features...")
    data = prepare_features(data)
    
    # Transformer la cible
    y = transform_target(data['trip_duration'])
    
    # Créer un pipeline de prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), FEATURES['numerical']),
            ('cat', OneHotEncoder(drop='first'), FEATURES['categorical'])
        ]
    )
    
    # Diviser les données
    print("\nDivision des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES['numerical'] + FEATURES['categorical']], 
        y, 
        **SPLIT_PARAMS
    )
    print(f"Données divisées: {X_train.shape[0]} exemples d'entraînement, {X_test.shape[0]} exemples de test")
    
    # Créer un pipeline complet
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', train_model())
    ])
    
    # Entraîner le modèle
    print("\nEntraînement du modèle...")
    pipeline.fit(X_train, y_train)
    
    # Évaluer le modèle
    print("\nÉvaluation du modèle...")
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    
    print(f"RMSE d'entraînement: {train_rmse:.4f}")
    print(f"RMSE de test: {test_rmse:.4f}")
    
    # Sauvegarder le modèle
    print("\nSauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    save_model(pipeline, "models/ridge_model.joblib")

if __name__ == "__main__":
    main() 
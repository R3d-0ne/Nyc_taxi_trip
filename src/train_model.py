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
    step1_add_features, 
    transform_target
)
from src.models.train import train_model, evaluate_model, save_model
from config.model_config import FEATURES, SPLIT_PARAMS, PATHS

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Préparer les features pour l'entraînement
    
    Args:
        df: DataFrame avec les données brutes
        
    Returns:
        DataFrame avec les features préparées
    """
    # Appliquer la fonction step1_add_features
    return step1_add_features(df)

def main():
    # Charger les données
    print("Chargement des données...")
    data = load_data(PATHS['data'])
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
    train_metrics = evaluate_model(pipeline, X_train, y_train)
    test_metrics = evaluate_model(pipeline, X_test, y_test)
    
    print(f"Métriques d'entraînement:")
    print(f"- RMSE: {train_metrics['rmse']:.4f}")
    print(f"- R²: {train_metrics['r2']:.4f}")
    
    print(f"\nMétriques de test:")
    print(f"- RMSE: {test_metrics['rmse']:.4f}")
    print(f"- R²: {test_metrics['r2']:.4f}")
    
    # Sauvegarder le modèle
    print("\nSauvegarde du modèle...")
    save_model(pipeline, PATHS['model'])

if __name__ == "__main__":
    main() 
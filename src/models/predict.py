"""
Module pour faire des prédictions avec le modèle entraîné
"""
import numpy as np
import pandas as pd
from ..preprocessing.preprocess import step1_add_features
from .train import load_model
from config.model_config import FEATURES, PATHS

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Préparer les features pour la prédiction
    
    Args:
        df: DataFrame avec les données brutes
        
    Returns:
        DataFrame avec les features préparées
    """
    # Appliquer la fonction step1_add_features
    data = step1_add_features(df)
    
    # Sélectionner les features utilisées pour l'entraînement
    features = FEATURES['numerical'] + FEATURES['categorical']
    
    return data[features]

def predict_duration(model, X: pd.DataFrame) -> np.ndarray:
    """
    Prédire la durée des trajets
    
    Args:
        model: Modèle entraîné
        X: DataFrame avec les features préparées
        
    Returns:
        Array des prédictions (en secondes)
    """
    # Faire les prédictions (log-transformées)
    y_pred_log = model.predict(X)
    
    # Convertir en secondes
    y_pred = np.expm1(y_pred_log)
    
    return y_pred

def format_predictions(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Formater les prédictions dans un DataFrame
    
    Args:
        df: DataFrame original
        predictions: Array des prédictions
        
    Returns:
        DataFrame avec les prédictions formatées
    """
    results = pd.DataFrame({
        'pickup_datetime': df['pickup_datetime'],
        'pickup_location': df.apply(lambda x: f"({x['pickup_latitude']:.4f}, {x['pickup_longitude']:.4f})", axis=1),
        'dropoff_location': df.apply(lambda x: f"({x['dropoff_latitude']:.4f}, {x['dropoff_longitude']:.4f})", axis=1),
        'predicted_duration': predictions.round(0).astype(int)
    })
    
    # Convertir la durée en format plus lisible
    results['predicted_duration_formatted'] = pd.to_timedelta(results['predicted_duration'], unit='s')
    
    return results 
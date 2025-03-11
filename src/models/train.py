"""
Module pour l'entraînement des modèles
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from config.model_config import MODEL_PARAMS

def train_model(X_train=None, y_train=None, params=None):
    """
    Créer et entraîner un modèle Ridge
    
    Args:
        X_train: Features d'entraînement 
        y_train: Cible d'entraînement 
        params: Paramètres du modèle (utilise MODEL_PARAMS par défaut)
        
    Returns:
        Modèle Ridge entraîné
    """
    if params is None:
        params = MODEL_PARAMS
        
    model = Ridge(**params)
    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    """
    Évaluer un modèle
    
    Args:
        model: Modèle à évaluer
        X: Features
        y: Vraies valeurs
        
    Returns:
        Dictionnaire des métriques
    """
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics

def save_model(model, filepath: str):
    """
    Sauvegarder un modèle
    
    Args:
        model: Modèle à sauvegarder
        filepath: Chemin où sauvegarder le modèle
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans {filepath}")

def load_model(filepath: str):
    """
    Charger un modèle
    
    Args:
        filepath: Chemin du modèle à charger
        
    Returns:
        Modèle chargé
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
    
    model = joblib.load(filepath)
    return model 
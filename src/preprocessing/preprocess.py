"""
Module pour le prétraitement des données
"""
import numpy as np
import pandas as pd

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extraire les caractéristiques temporelles de pickup_datetime
    
    Args:
        df: DataFrame contenant la colonne pickup_datetime
        
    Returns:
        DataFrame avec les nouvelles caractéristiques
    """
    data = df.copy()
    
    # Extraire les caractéristiques temporelles
    data['hour'] = data['pickup_datetime'].dt.hour
    data['weekday'] = data['pickup_datetime'].dt.weekday
    data['month'] = data['pickup_datetime'].dt.month
    
    data['abnormal_period'] = 0  
    
    return data

def transform_target(y: pd.Series) -> pd.Series:
    """
    Transformer la variable cible (log transformation)
    
    Args:
        y: Série contenant la variable cible
        
    Returns:
        Série transformée
    """
    return np.log1p(y) 
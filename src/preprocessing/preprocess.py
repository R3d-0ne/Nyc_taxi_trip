"""
Module pour le prétraitement des données
"""
import numpy as np
import pandas as pd
import datetime

# Les dates anormales sont calculées à partir des données,
abnormal_dates = None

def step1_add_features(X):
    res = X.copy()
    res['weekday'] = res['pickup_datetime'].dt.weekday
    res['month'] = res['pickup_datetime'].dt.month
    res['hour'] = res['pickup_datetime'].dt.hour
    res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int) if abnormal_dates is not None else 0
    return res

def transform_target(y: pd.Series) -> pd.Series:
    """
    Transformer la variable cible (log transformation)
    
    Args:
        y: Série contenant la variable cible
        
    Returns:
        Série transformée
    """
    return np.log1p(y) 
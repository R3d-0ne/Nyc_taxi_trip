"""
Module pour charger les données
"""
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """Charge les données depuis un ZIP ou un CSV."""
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("Aucun fichier CSV trouvé.")
            with zip_ref.open(csv_files[0]) as f:
                return pd.read_csv(f)
    except:
        return pd.read_csv(filepath)

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données et convertit la date."""
    from ..preprocessing import preprocess

    # Convertir d'abord en datetime
    data = data.assign(pickup_datetime=pd.to_datetime(data['pickup_datetime']))
    
    # Calculer le nombre de trajets par jour
    df_counts = data['pickup_datetime'].dt.date.value_counts()
    
    # Mettre à jour la variable abnormal_dates dans le module preprocess
    preprocess.abnormal_dates = df_counts[df_counts < 6300]
    return data.drop(columns=[c for c in ['id', 'dropoff_datetime'] if c in data.columns], errors='ignore')

def split_train_test(data: pd.DataFrame, target_col: str = 'trip_duration', test_size: float = 0.3, random_state: int = 42):
    """Divise les données en train et test."""
    return train_test_split(data.drop(columns=[target_col]), data[target_col], test_size=test_size, random_state=random_state)

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os
import numpy as np

import yaml


ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH =  CONFIG['paths']['data']
PROCESSED_PATH =  CONFIG['paths']['processed_path']
FEATURES =  CONFIG['ml']['features']
TEST_SIZE =  CONFIG['ml']['test_size']
RANDOM_STATE =  CONFIG['ml']['random_state']
ABNORMAL_PERIOD =  CONFIG['ml']['abnormal_period']


def preprocess():
    con = sqlite3.connect(DB_PATH)
    data = pd.read_sql('SELECT * FROM train', con)
    con.close()
    print(f"Prétraitement des données...")

    # Conversion de la date
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    
    # Création des features temporelles
    data['weekday'] = data['pickup_datetime'].dt.weekday
    data['month'] = data['pickup_datetime'].dt.month
    data['hour'] = data['pickup_datetime'].dt.hour
    
    # Identification des périodes anormales (moins de 6300 trajets)
    df_counts = data['pickup_datetime'].dt.date.value_counts()
    abnormal_dates = df_counts[df_counts < ABNORMAL_PERIOD]
    data['abnormal_period'] = data['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
    
    features = FEATURES
    X = data[features]
    y = data['target']
    
    y = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Fin du prétraitement des données")

    print(f"Sauvegarde des données dans {PROCESSED_PATH}...")

    model_dir = os.path.dirname(PROCESSED_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(PROCESSED_PATH, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Données prétraitées et enregistrées dans {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess() 
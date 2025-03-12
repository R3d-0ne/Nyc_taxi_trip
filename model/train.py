import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os, pickle

# Chemins absolus
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "data", "processed", "nyc_taxi.db")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ridge_model.joblib")
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "New_York_City_Taxi_Trip_Duration.zip")

def transform_target(y):
    """Transformer la variable cible (log transformation)"""
    return np.log1p(y)

def inverse_transform_target(y_log):
    """Transforme la cible prédite en échelle originale"""
    return np.expm1(y_log)

def preprocess_data(X):
    print(f"Preprocessing data")
    return X

def load_train_data(path):
    """Charger les données d'entraînement depuis SQLite"""
    print(f"Lecture des données depuis : {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    
    # Convertir pickup_datetime en datetime
    data_train['pickup_datetime'] = pd.to_datetime(data_train['pickup_datetime'])
    
    X = data_train.drop(columns=['target'])
    y = transform_target(data_train['target'])
    return X, y

def prepare_features(X):
    """Préparer les features de base"""
    X = X.copy()
    X['weekday'] = X['pickup_datetime'].dt.weekday
    X['month'] = X['pickup_datetime'].dt.month
    X['hour'] = X['pickup_datetime'].dt.hour
    
    # Calculer les dates anormales (moins de 6300 trajets)
    df_counts = X['pickup_datetime'].dt.date.value_counts()
    abnormal_dates = df_counts[df_counts < 6300]
    X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
    
    return X

def train_model(X, y):
    """Entraîner le modèle Ridge avec les features de base"""
    print("Construction du modèle...")
    
    num_features = ['abnormal_period', 'hour']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessing', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(X[train_features], y)
    y_pred = model.predict(X[train_features])
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE sur les données d'entraînement : {rmse:.4f}")
    
    return model, train_features

def persist_model(model, path):
    """Sauvegarder le modèle"""
    print(f"Sauvegarde du modèle dans {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(path, "wb") as file:
        pickle.dump(model, file)
    print("Terminé")

def save_to_sqlite(data, db_path, test_size=0.3, random_state=42):
    """Sauvegarde les données dans une base SQLite avec split train/test."""
    print(f"Préparation et sauvegarde des données dans {db_path}")
    
    # Préparer les données
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    X = data.drop(columns=['id', 'dropoff_datetime', 'trip_duration'])
    y = data['trip_duration']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Créer les DataFrames complets
    train_data = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
    test_data = pd.concat([X_test, pd.Series(y_test, name='target')], axis=1)
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Sauvegarder dans SQLite
    con = sqlite3.connect(db_path)
    train_data.to_sql('train', con, if_exists='replace', index=False)
    test_data.to_sql('test', con, if_exists='replace', index=False)
    con.close()
    
    print(f"Données sauvegardées : {len(train_data)} lignes train, {len(test_data)} lignes test")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("Première exécution : préparation des données...")
        data = pd.read_csv(RAW_DATA_PATH)
        save_to_sqlite(data, DB_PATH)
    
    X_train, y_train = load_train_data(DB_PATH)
    X_train = prepare_features(X_train)
    model, features = train_model(X_train, y_train)
    persist_model((model, features), MODEL_PATH)
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
import yaml

# Configuration
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = os.path.join(ROOT_DIR, CONFIG['paths']['data'])
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ridge_model.joblib")
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "New_York_City_Taxi_Trip_Duration.zip")
FEATURES = CONFIG['ml']['features']
TEST_SIZE = CONFIG['ml']['test_size']
RANDOM_STATE = CONFIG['ml']['random_state']
ABNORMAL_PERIOD = CONFIG['ml']['abnormal_period']


class CustomModel:
    def __init__(self, model_path=None, db_path=None):
        self.model_path = model_path if model_path else MODEL_PATH
        self.db_path = db_path if db_path else DB_PATH
        self.model = None
        self.features = None
        
    def load(self):
        """Charger le modèle depuis le fichier"""
        with open(self.model_path, 'rb') as f:
            self.model, self.features = pickle.load(f)
            
    def predict_and_store(self, input_datetime):
        """Faire une prédiction et stocker le résultat"""
        # Prétraitement
        temp_df = pd.DataFrame([{'pickup_datetime': input_datetime}])
        processed = prepare_features(temp_df)
        input_data = processed[self.features]
        
        # Prédiction
        prediction_log = self.model.predict(input_data)[0]
        duration_seconds = inverse_transform_target(prediction_log)
        duration_minutes = round(duration_seconds / 60, 1)
        
        # Stockage
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (pickup_datetime, predicted_duration)
                     VALUES (?, ?)''',
                  (input_datetime.isoformat(), duration_minutes))
        conn.commit()
        conn.close()
        
        return duration_minutes


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
    
    # Calculer les dates anormales
    df_counts = X['pickup_datetime'].dt.date.value_counts()
    abnormal_dates = df_counts[df_counts < ABNORMAL_PERIOD]
    X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
    
    return X


def train_model(X, y):
    """Entraîner le modèle Ridge avec les features de base"""
    print("Construction du modèle...")
    
    # Utiliser les features depuis la configuration
    train_features = FEATURES
    
    # Séparation des features numériques et catégorielles
    num_features = ['abnormal_period', 'hour']
    cat_features = ['weekday', 'month']

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessing', column_transformer),
        ('regression', Ridge(alpha=CONFIG['ml']['alpha']))
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


def save_to_sqlite(data, db_path, test_size=None, random_state=None):
    """Sauvegarde les données dans une base SQLite avec split train/test."""
    print(f"Préparation et sauvegarde des données dans {db_path}")
    
    # Utiliser les valeurs de configuration par défaut si non spécifiées
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    
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
    print(f"Modèle entraîné et sauvegardé dans {MODEL_PATH}")
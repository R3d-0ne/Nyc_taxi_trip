import pandas as pd
import sqlite3
import os
import yaml
import pickle
from sklearn.metrics import mean_squared_error

# Configuration
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Chemins des fichiers
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ridge_model_custom.joblib")

def load_test_data():
    db_path = os.path.join(ROOT_DIR, CONFIG['paths']['data'])
    con = sqlite3.connect(db_path)
    data = pd.read_sql('SELECT * FROM test LIMIT 5', con)
    con.close()
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    return data.drop(columns=['target']), data['target']

def load_model():
    print(f"Chargement du modèle personnalisé depuis {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Charger les données de test
    X_test, y_test = load_test_data()
    
    # Charger le modèle
    model = load_model()
    
    # Prédictions
    predictions = model.predict(X_test)
    
    # Affichage
    results = X_test.copy()
    results['true_duration'] = y_test
    results['predicted_duration'] = predictions.round(2)
    print("\nRésultats des prédictions :")
    print(results[['pickup_datetime', 'true_duration', 'predicted_duration']])
    print(f"\nMSE moyen : {mean_squared_error(y_test, predictions):.2f}")

if __name__ == "__main__":
    main() 
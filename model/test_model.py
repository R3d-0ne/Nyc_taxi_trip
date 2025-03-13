import mlflow
import pandas as pd
import sqlite3
import os
import yaml
from sklearn.metrics import mean_squared_error

# Configuration
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

MLFLOW_TRACKING_URI = "file:" + os.path.join(ROOT_DIR, CONFIG['paths']['mlruns'])
MODEL_NAME = CONFIG['mlflow']['model_name']

def load_test_data():
    con = sqlite3.connect(CONFIG['paths']['data'])
    data = pd.read_sql('SELECT * FROM test LIMIT 5', con)
    con.close()
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    return data.drop(columns=['target']), data['target']

def main():
    # Charger les données de test
    X_test, y_test = load_test_data()
    
    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Récupérer la dernière version du modèle
    client = mlflow.MlflowClient()
    model_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    
    if not model_versions:
        print("Aucun modèle trouvé dans le registre")
        return
    
    latest_version = model_versions[0].version
    print(f"Chargement du modèle {MODEL_NAME}, version {latest_version}")
    
    # Charger le modèle
    model_uri = f"models:/{MODEL_NAME}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    
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
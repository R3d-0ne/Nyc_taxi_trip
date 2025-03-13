import pickle
import pandas as pd
import mlflow
import yaml

# Configuration
with open("config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

MLFLOW_TRACKING_URI = "file:" + CONFIG['paths']['mlruns']
MODEL_NAME = CONFIG['mlflow']['model_name']
PROCESSED_PATH = CONFIG['paths']['processed_path']

def load_test_data():
    with open(PROCESSED_PATH, "rb") as file:
        _, X_test, _, y_test = pickle.load(file)
    return X_test, y_test

if __name__ == "__main__":
    # Charger les données de test
    X_test, y_test = load_test_data()
    X_test = X_test[:5]
    y_test = y_test[:5]

    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Charger le dernier modèle du registre
    mlflow_client = mlflow.MlflowClient()
    model_metadata = mlflow_client.get_latest_versions(MODEL_NAME, stages=["None"])
    latest_model_version = model_metadata[0].version

    print("Load model from the model registry")
    model_uri = f"models:/{MODEL_NAME}/{latest_model_version}"
    print(f"Model URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    y_pred = model.predict(X_test)

    # Afficher les résultats
    data_test = X_test.copy()
    data_test['target-true'] = y_test
    data_test['target-pred'] = y_pred
    print(data_test)
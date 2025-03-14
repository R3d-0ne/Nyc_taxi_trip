import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import os
import yaml

# Configuration
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
config_path = os.path.join(ROOT_DIR, "config.yml")

with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = os.path.join(ROOT_DIR, CONFIG['paths']['data'])
MLFLOW_TRACKING_URI = "file:" + os.path.join(ROOT_DIR, CONFIG['paths']['mlruns'])
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_name']
MODEL_NAME = CONFIG['mlflow']['model_name']
ARTIFACT_PATH = CONFIG['mlflow']['artifact_path']
ALPHAS = [0.01, 0.1, 1, 10]
RUN_NAME = "ridge_regression"
PROCESSED_PATH = os.path.join(ROOT_DIR, CONFIG['paths']['processed_path'])


def load_train_data():
    with open(PROCESSED_PATH, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle Ridge, calcule les métriques et enregistre le tout dans MLflow
    """
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Enregistrement des métriques
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Enregistrement du modèle
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=ARTIFACT_PATH,
        signature=signature
    )
    
    # Retourne les métriques pour comparaison
    metrics = {"root_mean_squared_error": rmse, "r2_score": r2}
    return metrics

def main():
    # Configuration de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Chargement des données
    X_train, X_test, y_train, y_test = load_train_data()
    
    # Affichage des informations sur les données
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Recherche des valeurs NaN
    # if isinstance(X_train, np.ndarray):
    #     print(f"NaN in X_train: {np.isnan(X_train).any()}")
    #     print(f"NaN in X_test: {np.isnan(X_test).any()}")
    
    # Variables pour suivre le meilleur modèle
    best_score = float('inf')
    best_run_id = None
    
    # Exécution des expériences pour différentes valeurs d'alpha
    with mlflow.start_run(run_name=RUN_NAME) as parent_run:
        for i, alpha in enumerate(ALPHAS, 1):
            print(f"\n***** ITERATION {i} from {len(ALPHAS)} *****")
            
            # Création du modèle Ridge avec l'alpha actuel
            model = Ridge(alpha=alpha)
            
            # Entraînement et évaluation dans une sous-expérience
            with mlflow.start_run(run_name=f"{RUN_NAME}_{i:02}", nested=True) as child_run:
                # Enregistrement du paramètre alpha
                mlflow.log_param("alpha", alpha)
                
                # Entraînement et évaluation du modèle
                metrics = train_and_log_model(model, X_train, X_test, y_train, y_test)
                
                # Mise à jour du meilleur modèle si nécessaire
                if metrics["root_mean_squared_error"] < best_score:
                    best_score = metrics["root_mean_squared_error"]
                    best_run_id = child_run.info.run_id
                
                # Affichage des résultats
                print(f"rmse: {metrics['root_mean_squared_error']}")
                print(f"r2: {metrics['r2_score']}")
        
        # Enregistrement du meilleur modèle dans le registre MLflow
        print("#" * 20)
        if best_run_id:
            model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
            mv = mlflow.register_model(model_uri, MODEL_NAME)
            print("Modèle enregistré dans le registre:")
            print(f"Nom: {mv.name}")
            print(f"Version: {mv.version}")
            print(f"Source: {mv.source}")
        else:
            print("Aucun modèle n'a pu être enregistré.")

if __name__ == "__main__":
    main()
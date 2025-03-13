import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.models import infer_signature
import os
import yaml

# Configuration
with open("config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DB_PATH =  CONFIG['paths']['data']
MLFLOW_TRACKING_URI = "file:" + os.path.join(ROOT_DIR, CONFIG['paths']['mlruns'])
EXPERIMENT_NAME = CONFIG['mlflow']['experiment_name']
MODEL_NAME = CONFIG['mlflow']['model_name']
ARTIFACT_PATH = CONFIG['mlflow']['artifact_path']
ALPHAS = [0.01, 0.1, 1, 10]
RUN_NAME = "ridge_regression"
PROCESSED_PATH = CONFIG['paths']['processed_path']


def load_train_data():
    with open(PROCESSED_PATH, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    
    signature = infer_signature(X_train, y_train)
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=ARTIFACT_PATH,
        signature=signature
    )

    results = mlflow.evaluate(
        model_info.model_uri,
        data=pd.concat([X_test, y_test], axis=1),
        targets=y_test.name,
        model_type="regressor",
        evaluators=["default"]
    )
    return results

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    X_train, X_test, y_train, y_test = load_train_data()

    mlflow.set_experiment(EXPERIMENT_NAME)
    
    best_score = float('inf')
    best_run_id = None
    
    num_iterations = len(ALPHAS)
    k = 0
    run_name = "ridge_regression"

    with mlflow.start_run(run_name=run_name, description=run_name) as parent_run:
        for alpha in ALPHAS:
            k += 1
            print(f"\n***** ITERATION {k} from {num_iterations} *****")
            child_run_name = f"{run_name}_{k:02}"
            model = Ridge(alpha=alpha)
            
            with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                mlflow.log_param("alpha", alpha)
                results = train_and_log_model(model, X_train, X_test, y_train, y_test)
                
                if results.metrics['root_mean_squared_error'] < best_score:
                    best_score = results.metrics['root_mean_squared_error']
                    best_run_id = child_run.info.run_id
                
                print(f"rmse: {results.metrics['root_mean_squared_error']}")
                print(f"r2: {results.metrics['r2_score']}")
        
        print("#" * 20)
        model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        print("Model saved to the model registry:")
        print(f"Name: {mv.name}")
        print(f"Version: {mv.version}")
        print(f"Source: {mv.source}")

if __name__ == "__main__":
    main()
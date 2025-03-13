# NYC Taxi Trip Duration Prediction

API de prédiction de la durée des trajets taxis à New York

## MLflow Tracking

Le projet utilise MLflow pour suivre les expériences et gérer les modèles.

### Structure
- Les runs sont organisés avec une structure parent/child
- Chaque run child teste un hyperparamètre alpha différent
- Le meilleur modèle est automatiquement enregistré dans le registry

### Utilisation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancer l'entraînement :
```bash
python src/train_taxi_mlflow.py
```

3. Tester le modèle :
```bash
python src/test_taxi_mlflow.py
```

4. Visualiser les résultats :
```bash
mlflow ui
```
Accéder à http://localhost:5000

5. Démarrer l'API :
```bash
uvicorn src.api.main:app --reload
```

6. Accéder à la documentation :
http://localhost:8000/docs

## Exemple de requête
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"pickup_datetime": "2024-03-15T14:30:00"}'
```

## Réponse attendue
```json
{
  "duree_trajet_minutes": 25.5,
  "heure_arrivee": "2024-03-15T14:55:30"
}
```

## Structure de la base de données
Les prédictions sont stockées dans `data/processed/predictions.db` :

| Colonne                | Type         | Description                          |
|------------------------|--------------|--------------------------------------|
| id                     | INTEGER      | Clé primaire auto-incrémentée        |
| pickup_datetime        | TEXT         | Date/heure de prise en charge (ISO)  |
| predicted_duration     | REAL         | Durée prédite en minutes             |
| prediction_timestamp   | DATETIME     | Horodatage de la prédiction          |

### Métriques suivies
- RMSE (Root Mean Squared Error)
- R² Score
- MSE (Mean Squared Error)
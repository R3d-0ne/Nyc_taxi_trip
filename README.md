# NYC Taxi Trip Duration Prediction

API de prédiction de la durée des trajets taxis à New York

## Utilisation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Entraîner le modèle :
```bash
python model/train.py
```

3. Démarrer l'API :
```bash
uvicorn model.api.main:app --reload
```

4. Accéder à la documentation :
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
```

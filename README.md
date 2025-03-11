# Prédiction de la durée des trajets en taxi à New York

Ce projet vise à prédire la durée des trajets en taxi à New York en utilisant des techniques de Machine Learning.

## Structure du projet

```
├── config/             # Fichiers de configuration
├── data/              # Données
│   ├── raw/          # Données brutes
│   └── processed/    # Données prétraitées
├── notebooks/         # Notebooks Jupyter
└── src/              # Code source
    ├── data/         # Scripts de gestion des données
    ├── models/       # Scripts des modèles
    ├── preprocessing/ # Scripts de prétraitement
    └── utils/        # Utilitaires
```

## Installation

```bash
pip install -r requirements.txt
```

## Téléchargement des données

Pour télécharger les données, exécutez la commande suivante :

```bash
wget https://github.com/eishkina-estia/ML2023/raw/main/data/New_York_City_Taxi_Trip_Duration.zip -P data/raw
```

Le fichier sera téléchargé dans le dossier `data/raw` du projet. Le chemin complet sera :
```
data/raw/New_York_City_Taxi_Trip_Duration.zip
```

## Entraînement du modèle

Pour entraîner le modèle, exécutez la commande suivante :

```bash
python -m src.train_model
```

Cette commande va :
1. Charger les données depuis le fichier ZIP
2. Préparer les features (caractéristiques temporelles)
3. Entraîner un modèle Ridge avec un pipeline de prétraitement
4. Évaluer le modèle sur les données d'entraînement et de test
5. Sauvegarder le modèle dans le dossier `models/`

## Test d'inférence

Pour tester le modèle sur quelques exemples, exécutez :

```bash
python -m src.test_inference
```

Cette commande va :
1. Charger les données et le modèle entraîné
2. Sélectionner aléatoirement 5 exemples
3. Préparer les features pour ces exemples
4. Faire des prédictions de durée de trajet
5. Afficher les résultats

## Features utilisées

Le modèle utilise les features suivantes :
- **Features numériques** : 
  - Heure du jour (`hour`)
  - Indicateur de période anormale (`abnormal_period`)
- **Features catégorielles** : 
  - Jour de la semaine (`weekday`)
  - Mois (`month`)

Ces features sont prétraitées avant d'être utilisées par le modèle :
- Les features numériques sont standardisées avec `StandardScaler`
- Les features catégorielles sont encodées avec `OneHotEncoder`

## Performance du modèle

Le modèle Ridge entraîné atteint une performance de :
- RMSE sur l'ensemble d'entraînement : ~0.7925 (log)
- RMSE sur l'ensemble de test : ~0.7935 (log)

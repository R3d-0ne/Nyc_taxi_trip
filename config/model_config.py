"""
Configuration du modèle et des chemins
"""

# Chemins des fichiers
PATHS = {
    'data': 'data/raw/New_York_City_Taxi_Trip_Duration.zip',
    'model': 'models/ridge_model.joblib'
}

# Paramètres du modèle Ridge (section 4.6)
MODEL_PARAMS = {
    'alpha': 1.0,  # Paramètre de régularisation
    'random_state': 42,  # Pour la reproductibilité
    'solver': 'auto'  # Solveur automatique
}

# Features à utiliser (section 4.6)
FEATURES = {
    'numerical': [
        'abnormal_period',  # Indicateur de période anormale (0 ou 1)
        'hour'             # Heure du jour (0-23)
    ],
    'categorical': [
        'weekday',         # Jour de la semaine (0-6)
        'month'           # Mois (1-12)
    ]
}

# Paramètres de division des données
SPLIT_PARAMS = {
    'test_size': 0.2,     # 20% pour le test
    'random_state': 42    # Pour la reproductibilité
} 
"""
Configuration du modèle
"""

# Paramètres du modèle Ridge
MODEL_PARAMS = {
    'alpha': 1.0,  # Paramètre de régularisation
    'random_state': 42,  # Pour la reproductibilité
    'solver': 'auto'  # Solveur automatique
}

# Features à utiliser
FEATURES = {
    'numerical': ['abnormal_period', 'hour'],
    'categorical': ['weekday', 'month']
}

# Paramètres de division des données
SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42
} 
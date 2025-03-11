"""
Script pour tester l'inférence du modèle
"""
from src.data.load_data import load_data, prepare_data
from src.models.predict import prepare_features, predict_duration, format_predictions
from src.models.train import load_model

def main():
    # Charger les données de test
    print("Chargement des données...")
    data = load_data("data/raw/New_York_City_Taxi_Trip_Duration.zip")
    data = prepare_data(data)
    
    # Prendre un petit échantillon pour le test
    test_data = data.sample(n=5, random_state=42)
    
    # Charger le modèle
    print("\nChargement du modèle...")
    model = load_model("models/ridge_model.joblib")
    
    # Préparer les features
    print("\nPréparation des features...")
    X = prepare_features(test_data)
    
    # Faire les prédictions
    print("\nPrédiction des durées...")
    predictions = predict_duration(model, X)
    
    # Formater les résultats
    results = format_predictions(test_data, predictions)
    
    # Afficher les résultats
    print("\nRésultats des prédictions :")
    print(results[['pickup_datetime', 'pickup_location', 'dropoff_location', 'predicted_duration_formatted']])

if __name__ == "__main__":
    main() 
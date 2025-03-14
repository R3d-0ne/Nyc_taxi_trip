import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import json
import os
from PIL import Image

# Configuration API
API_URL = "http://localhost:8000"  # Adresse de votre API FastAPI

# Configuration du chemin vers l'image
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
image_path = os.path.join(ROOT_DIR, "data", "images", "image.webp")

# Affichage de l'image centrée
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Vérifier si l'image existe
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="", use_container_width=True)
    else:
        st.error(f"Image non trouvée: {image_path}")

# st.title("New York Yankees Cab")

st.sidebar.title("New York Yankees Cab")

# Affichage du logo des Yankees dans la sidebar
logo_url = "https://upload.wikimedia.org/wikipedia/commons/3/35/Yankees_logo.png"
st.sidebar.image(logo_url, width=100, caption="New York Yankees")

# Le reste de votre sidebar
page = st.sidebar.selectbox("Navigation", ["Prédictions", "Historique", "Statistiques"])

if page == "Prédictions":
    st.header("Faire une nouvelle prédiction")
    
    # Formulaire de saisie
    pickup_date = st.date_input("Date de prise en charge", datetime.now().date())
    pickup_time = st.time_input("Heure de prise en charge", datetime.now().time())
    pickup_datetime = datetime.combine(pickup_date, pickup_time)
    
    model_type = st.radio("Modèle à utiliser", ["Standard", "Personnalisé"])
    
    if st.button("🔮 Prédire"):
        # Préparation des données pour l'API
        input_data = {
            "pickup_datetime": pickup_datetime.isoformat()
        }
        
        # Appel à l'API
        if model_type == "Standard":
            endpoint = f"{API_URL}/predict"
        else:
            endpoint = f"{API_URL}/predict_custom"
            
        try:
            with st.spinner("Calcul en cours..."):
                response = requests.post(endpoint, json=input_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if model_type == "Standard":
                        st.success(f"Durée estimée du trajet: {result['duree_trajet_minutes']} minutes")
                        st.info(f"Heure d'arrivée approximative: {result['heure_arrivee approximative']}")
                    else:
                        st.success(f"Durée estimée du trajet: {result['duree_trajet_secon']} secondes")
                else:
                    st.error(f"Erreur API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
            st.info("Assurez-vous que l'API FastAPI est en cours d'exécution (uvicorn src.api.main:app --reload)")

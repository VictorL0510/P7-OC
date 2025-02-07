import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💰",
    layout="wide"
)

# URL de l'API (à modifier selon votre déploiement)
API_URL = "https://credit-scoring-api-320550565176.europe-west1.run.app"

def load_data():
    """Charge les données de test"""
    df = pd.read_csv('test_df.csv',sep=';')
    return df

def get_available_models():
    """Récupère la liste des modèles disponibles"""
    try:
        response = requests.get(f"{API_URL}/available_models")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def load_model(model_name):
    """Charge un modèle spécifique"""
    try:
        response = requests.post(
            f"{API_URL}/load_model_by_name",
            json={"name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_client(features):
    """Fait une prédiction pour un client"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print("Error details:", str(e))
        print("Error type:", type(e).__name__)
        return None

def create_gauge_chart(value, threshold):
    """Crée une jauge pour visualiser le risque"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold * 100], 'color': "lightgreen"},
                {'range': [threshold * 100, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        },
        number={'suffix': "%"}
    ))
    
    fig.update_layout(
        title={
            'text': "Probabilité de Défaut",
            'y':0.8,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

# Sidebar pour la sélection du modèle
st.sidebar.header("🤖 Sélection du Modèle")

# Récupérer la liste des modèles
models = get_available_models()

if models:
    # Créer une liste de noms de modèles
    model_names = [model["name"] for model in models["models"]]
    
    # Sélectionner un modèle
    selected_model = st.sidebar.selectbox(
        "Choisissez un modèle",
        model_names
    )
    
    if selected_model:
        # Charger le modèle sélectionné
        model_info = load_model(selected_model)
        if model_info:
            st.sidebar.success(f"✅ {model_info['message']}")
            
            # Afficher le seuil
            st.sidebar.metric(
                "Seuil de décision",
                f"{model_info['threshold']:.2f}"
            )
else:
    st.sidebar.warning("⚠️ Aucun modèle disponible")

st.title("💰 Dashboard de Scoring Crédit")
st.markdown("""
Ce dashboard permet d'évaluer la probabilité de défaut de paiement d'un client
et d'obtenir une recommandation pour l'octroi du crédit.
""")

# Charger les données
try:
    df = load_data()

    # Sélection du client
    client_id = st.selectbox(
        "Sélectionner un client",
        df.index.tolist()
    )
    
    if st.button("Prédire") and client_id is not None:
        if not selected_model:
            st.warning("Veuillez d'abord sélectionner un modèle!")
        else:
            # Récupérer les features du client
            client_features = df.loc[client_id].to_dict()
            
            # Faire la prédiction
            prediction = predict_client(client_features)
            
            if prediction:
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Résultat de la prédiction")
                    if prediction["prediction"] == 0:
                        st.success("✅ Crédit Recommandé")
                    else:
                        st.error("❌ Crédit Non Recommandé")
                    
                    st.metric(
                        "Probabilité de Défaut",
                        f"{prediction['default_probability']:.1%}"
                    )
                    st.metric(
                        "Seuil de Décision",
                        f"{model_info['threshold']:.1%}"
                    )
                
                with col2:
                    st.subheader("Visualisation du Risque")
                    fig = create_gauge_chart(
                        prediction["default_probability"],
                        model_info['threshold']
                    )
                    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")

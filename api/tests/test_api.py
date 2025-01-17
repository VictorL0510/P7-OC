import pytest
import requests
import pandas as pd
import numpy as np
import pickle
import os

# URL de base de l'API - utiliser une variable d'environnement avec fallback
BASE_URL = os.getenv('API_URL', 'http://localhost:8000')

# URL de base de l'API
# BASE_URL = "http://localhost:8000"  # Le port externe sur lequel le container est mappé

def get_model_features():
    """Récupères la liste des features attendues par le modèle"""
    model_path = os.path.join('models', 'sk-learn-log-reg-model', 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model.feature_names_in_

@pytest.fixture
def test_data():
    """Charge les données de test depuis le CSV"""
    df = pd.read_csv('tests/sample_data.csv', sep=';')
    
    # Récupérer les features attendues par le modèle
    expected_features = get_model_features()
    print("\nFeatures attendues par le modèle:", expected_features)
    print("\nFeatures présentes dans le CSV:", df.columns.tolist())
    
    # Remplacer les valeurs NaN par 0
    df = df.fillna(0)
    
    # S'assurer que toutes les features nécessaires sont présentes
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        print(f"\nFeatures manquantes: {missing_features}")
        # Ajouter les features manquantes avec des valeurs par défaut
        for feature in missing_features:
            df[feature] = 0
    
    # Réorganiser les colonnes dans le même ordre que le modèle
    df = df[expected_features]
    
    # Prend la première ligne comme exemple
    sample = df.iloc[0].to_dict()
    # Convertir les numpy.int64 et numpy.float64 en types Python natifs
    sample = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in sample.items()}
    return {"features": sample}

def test_health_check():
    """Test de la route health"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)

def test_load_model_invalid():
    """Test du chargement d'un modèle invalide"""
    response = requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "modele_inexistant"}
    )
    assert response.status_code == 500  # L'API renvoie 500 pour les erreurs de chargement
    assert "erreur" in response.json()["detail"].lower()

def test_load_model_valid():
    """Test du chargement d'un modèle valide"""
    response = requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    assert response.status_code == 200
    assert "chargé avec succès" in response.json()["message"]

def test_predict_without_model(test_data):
    """Test de prédiction sans modèle chargé"""
    # S'assurer qu'aucun modèle n'est chargé
    response = requests.post(f"{BASE_URL}/unload_model")
    assert response.status_code == 200
    
    # Vérifier que le modèle est bien déchargé via /health
    health_response = requests.get(f"{BASE_URL}/health")
    assert health_response.status_code == 200
    assert health_response.json()["model_loaded"] == False
    
    # Tenter une prédiction sans modèle
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 400
    assert "Aucun modèle n'est chargé" in response.json()["detail"]

def test_predict_with_invalid_features():
    """Test de prédiction avec des features invalides"""
    # Charger d'abord un modèle
    requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    
    # Données de test invalides
    test_data = {
        "features": {
            "FEATURE_INVALIDE": 0
        }
    }
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 500  # L'API renvoie 500 pour les erreurs de prédiction
    assert "erreur" in response.json()["detail"].lower()

def test_predict_valid(test_data):
    """Test de prédiction avec des données réelles du CSV"""
    # Charger d'abord un modèle
    requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in [0, 1]
    assert "default_probability" in data
    assert "threshold" in data

def test_predict_batch():
    """Test de prédiction sur plusieurs lignes du CSV"""
    # Charger d'abord un modèle
    requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    
    # Charger plusieurs lignes du CSV
    df = pd.read_csv('tests/sample_data.csv', sep=';')
    
    # Récupérer les features attendues par le modèle
    expected_features = get_model_features()
    
    # Ajouter les features manquantes avec des valeurs par défaut
    missing_features = set(expected_features) - set(df.columns)
    for feature in missing_features:
        df[feature] = 0
    
    # Réorganiser les colonnes dans le même ordre que le modèle
    df = df[expected_features]
    
    df = df.fillna(0)  # Remplacer les NaN par 0
    
    # Tester les 5 premières lignes
    for idx in range(min(5, len(df))):
        row_dict = df.iloc[idx].to_dict()
        # Convertir les types numpy en types Python natifs
        row_dict = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in row_dict.items()}
        sample = {"features": row_dict}
        response = requests.post(f"{BASE_URL}/predict", json=sample)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]  # Correction de la syntaxe
        assert "default_probability" in data
        assert "threshold" in data
        assert response.elapsed.total_seconds() < 1.0  # La réponse doit être < 1 seconde

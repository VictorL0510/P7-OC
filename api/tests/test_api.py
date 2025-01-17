import pytest
import requests
import pandas as pd
import numpy as np
import os

# URL de base de l'API - utiliser une variable d'environnement avec fallback
BASE_URL = os.getenv('API_URL', 'http://localhost:8000')

@pytest.fixture
def test_data():
    """Charge les données de test depuis le CSV"""
    df = pd.read_csv('tests/sample_data.csv', sep=';')
    
    # Prendre la première ligne comme exemple
    test_data = {"features": df.iloc[0].to_dict()}
    return test_data

def test_health_check():
    """Test de l'endpoint health check"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_load_model_invalid():
    """Test de chargement d'un modèle invalide"""
    response = requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "invalid_model"}
    )
    assert response.status_code == 500

def test_load_model_valid():
    """Test de chargement d'un modèle valide"""
    response = requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    assert response.status_code == 200
    assert "chargé avec succès" in response.json()["message"]

def test_predict_valid(test_data):
    """Test de prédiction avec des données valides"""
    # Charger d'abord un modèle
    requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "default_probability" in result
    assert "threshold" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["default_probability"], float)
    assert isinstance(result["threshold"], float)

def test_predict_with_invalid_features():
    """Test de prédiction avec des features invalides"""
    # Charger d'abord un modèle
    requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    
    # Données de test invalides
    invalid_data = {
        "features": {
            "INVALID_FEATURE": 1,
            "ANOTHER_INVALID": "test"
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    assert response.status_code == 500
    assert "Erreur lors de la prédiction" in response.json()["detail"]

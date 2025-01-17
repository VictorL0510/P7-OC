import pytest
import requests
import pandas as pd
import numpy as np
import os

# URL de base de l'API - utiliser une variable d'environnement avec fallback
BASE_URL = os.getenv('API_URL', 'http://localhost:8000')

# Liste des features attendues par le modèle
EXPECTED_FEATURES = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL',
    'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE',
    'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE'
]

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
    assert response.status_code == 404

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
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "default_probability" in response.json()
    assert "threshold" in response.json()

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

def test_predict_batch():
    """Test de prédiction sur plusieurs lignes du CSV"""
    # Charger d'abord un modèle
    requests.post(
        f"{BASE_URL}/load_model_by_name",
        json={"name": "sk-learn-log-reg-model"}
    )
    
    # Charger plusieurs lignes du CSV
    df = pd.read_csv('tests/sample_data.csv', sep=';')
    
    # Prendre les 3 premières lignes
    batch_data = {"features": df.iloc[0:3].to_dict('records')}
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=batch_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert "probabilities" in result
    assert len(result["predictions"]) == 3
    assert len(result["probabilities"]) == 3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import mlflow
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Constante pour le dossier des modèles
MODELS_DIR = "models"

# Variables globales pour stocker le modèle chargé
model = None
model_threshold = None

class PredictionInput(BaseModel):
    features: Dict[str, Any]

class ModelName(BaseModel):
    name: str

def predict_class(probability: float, threshold: float) -> int:
    """Fait une prédiction en utilisant le seuil optimal du modèle"""
    return 1 if probability >= threshold else 0

@app.get("/available_models")
async def get_available_models():
    """Liste les modèles disponibles dans le dossier models"""
    try:
        models = []
        if os.path.exists(MODELS_DIR):
            for model_name in os.listdir(MODELS_DIR):
                model_path = os.path.join(MODELS_DIR, model_name)
                if os.path.isdir(model_path):
                    models.append({
                        "name": model_name,
                        "path": model_path
                    })
        
        return {
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des modèles: {str(e)}"
        )

@app.post("/load_model_by_name")
async def load_model_by_name(model_info: ModelName):
    """Charge le modèle spécifié depuis le dossier models"""
    global model, model_threshold
    
    try:
        model_path = os.path.join(MODELS_DIR, model_info.name)
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Modèle {model_info.name} non trouvé dans le dossier models"
            )
        
        # Charger le modèle
        model = mlflow.sklearn.load_model(model_path)
        
        # Charger le seuil depuis le fichier
        threshold_path = os.path.join(model_path, "threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                model_threshold = float(f.read().strip())
        else:
            # Seuil par défaut si le fichier n'existe pas
            model_threshold = 0.48
        
        return {
            "message": f"Modèle {model_info.name} chargé avec succès",
            "threshold": model_threshold
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement du modèle: {str(e)}"
        )

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Fait une prédiction avec le modèle chargé"""
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Aucun modèle n'est chargé. Utilisez /load_model_by_name d'abord."
        )
    
    try:
        # Convertir les features en DataFrame
        features_df = pd.DataFrame([input_data.features])
        
        # Faire la prédiction
        proba = model.predict_proba(features_df)
        default_probability = float(proba[0][1])
        
        # Prédire la classe en utilisant le seuil
        prediction = predict_class(default_probability, model_threshold)
        
        return {
            "prediction": int(prediction),
            "default_probability": default_probability,
            "threshold": model_threshold
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.get("/health")
async def health():
    """Vérifie l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": model_threshold
    }

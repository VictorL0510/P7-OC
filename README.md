# Projet 7 OpenClassrooms : Implémentez un modèle de scoring

## Description du Projet
Ce projet consiste à développer un modèle de scoring pour une société financière qui souhaite mettre en œuvre un outil de "scoring crédit" pour calculer la probabilité qu'un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé.

## Structure du Projet

Le projet est organisé en plusieurs phases :

1. **Phase 1 : Notebook d'Analyse et Modélisation**
   - Analyse exploratoire des données (EDA)
   - Feature engineering
   - Entraînement de différents modèles
   - Optimisation du score business
   - Suivi des expériences avec MLflow

2. **Phase 2 : API de Prédiction**
   - Développement d'une API FastAPI
   - Déploiement sur Google Cloud Run
   - Tests unitaires et intégration continue

3. **Phase 3 : Dashboard Streamlit** (à venir)
   - Interface utilisateur interactive
   - Visualisation des prédictions
   - Explications des décisions du modèle

## Structure du Code

- `api/` : Contient l'API FastAPI pour le scoring crédit
  - `api.py` : Code principal de l'API
  - `Dockerfile` : Configuration pour la conteneurisation de l'API
  - `requirements.txt` : Dépendances Python pour l'API

- `models/` : Contient les modèles entraînés et leurs seuils
- `notebooks/` : Notebooks Jupyter pour l'analyse et l'entraînement des modèles

## Installation et Utilisation

Pour cette première phase :
```bash
# Cloner le repository
git clone https://github.com/VictorL0510/P7-OC.git

# Installer les dépendances
pip install -r requirements.txt

# Lancer MLflow UI (optionnel)
mlflow ui --port 5001
```

### API FastAPI

1. Construire l'image Docker :
```bash
cd api
docker build -t credit-scoring-api .
```

2. Démarrer le conteneur en local :
```bash
docker run -d -p 8000:8080 --name credit-scoring-api credit-scoring-api
```

L'API sera accessible sur http://localhost:8000

## Utilisation de l'API

L'API expose deux endpoints principaux :

- `/load_model_by_name` : Charge un modèle spécifique
  ```json
  POST /load_model_by_name
  {
    "name": "model"
  }
  ```

- `/predict` : Fait une prédiction avec le modèle chargé
  ```json
  POST /predict
  {
    "features": {
      "feature1": value1,
      "feature2": value2,
      ...
    }
  }
  ```

## Déploiement

L'API peut être déployée sur Google Cloud Run :

```bash
gcloud run deploy api --image europe-west1-docker.pkg.dev/[PROJECT_ID]/credit-scoring/api:latest --region europe-west1 --platform managed --allow-unauthenticated
```

## Technologies Utilisées
- Python
- Pandas, NumPy
- Scikit-learn
- MLflow
- Jupyter Notebook
- FastAPI
- Docker
- Google Cloud Run
- Scikit-learn
- XGBoost

## Auteur
Victor LESAFFRE

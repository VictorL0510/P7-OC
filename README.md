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

2. **Phase 2 : API de Prédiction** (à venir)
   - Développement d'une API FastAPI
   - Déploiement sur Google Cloud Run
   - Tests unitaires et intégration continue

3. **Phase 3 : Dashboard Streamlit** (à venir)
   - Interface utilisateur interactive
   - Visualisation des prédictions
   - Explications des décisions du modèle

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

## Technologies Utilisées
- Python
- Pandas, NumPy
- Scikit-learn
- MLflow
- Jupyter Notebook

## Auteur
Victor LESAFFRE

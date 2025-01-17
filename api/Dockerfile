FROM python:3.11

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de l'API et les tests
COPY api/api.py .
COPY api/tests/ ./tests/

# Créer le dossier models et copier son contenu
COPY api/models/ ./models/

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}

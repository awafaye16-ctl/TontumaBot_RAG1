# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile TontumaBot V3
# Ce conteneur embarque l'application FastAPI et tous ses modèles.
# Il est destiné au déploiement sur un serveur cloud (Render, Railway, Fly.io, VPS, ...).
# ─────────────────────────────────────────────────────────────────────────────

# ── Étape 1 : image de base Python 3.11 (légère) ─────────────────────────────
FROM python:3.11-slim



# ── Étape 2 : variables d'environnement ──────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Désactive TensorFlow/Keras pour accélérer le chargement
ENV USE_TF=0
ENV USE_TORCH=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TOKENIZERS_PARALLELISM=false


# definir le repertoire de travail dans docker

WORKDIR /app

# ── Étape 3 : installation des dépendances système ───────────────────────────
# ffmpeg : décodage audio pour le STT
# libsndfile1 : lecture/écriture audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Étape 4 : copie et installation des dépendances Python ───────────────────
COPY requirements.txt /app
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 --prefer-binary -r requirements.txt
# ici on installe les dépendances Python listées dans requirements.txt et le --no-cache-dir permet de ne pas stocker les fichiers temporaires pour réduire la taille de l'image Docker

# ── Étape 5 : copie du code source et des ressources ─────────────────────────
COPY . /app

# ── Étape 6 : création des dossiers utiles pour l'application afin de les rendre accessibles pour la lecture et l'écriture de fichiers ─────────────────────────────
RUN mkdir -p /app/data/documents /app/uploads /app/static /app/data/hf_cache

# ── Étape 7 : port exposé ────────────────────────────────────────────────────
# commande pour executer le conteneur et exposer le port 8008 pour l'application FastAPI
EXPOSE 8008

# ── Étape 8 : commande de démarrage ──────────────────────────────────────────
# Uvicorn lance l'application FastAPI. Le port est configurable via .env.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]
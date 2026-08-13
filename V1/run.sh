#!/usr/bin/env bash
# Lance le serveur TontumaBot dans son environnement virtuel.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Création de l'environnement virtuel..."
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt python-multipart
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "⚠️  Fichier .env créé — renseignez vos clés API (GROQ_API_KEY / GEMINI_API_KEY)."
fi

source .venv/bin/activate
PORT="${PORT:-8001}"
echo "🚀 TontumaBot démarré sur http://localhost:${PORT}"
exec python app.py
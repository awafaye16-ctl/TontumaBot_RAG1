# TONTOUMA-BOT — Borne_V1

Assistant hospitalier multimodal : gestion sécurisée des sessions conversationnelles avec Redis.

## Stack

- Python 3.11+, FastAPI, Pydantic v2, redis-py asynchrone
- Redis 7, Docker, pytest, httpx

## Installation locale

```bash
cd Borne_V1
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
cp .env.example .env            # puis éditer API_KEY
```

## Lancement avec Docker Compose

```bash
docker compose up --build
```

L'API écoute sur `http://localhost:8000`, Redis reste interne à Docker.

## Documentation OpenAPI

- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

## Exemples curl

Remplacer `$KEY` par la valeur de `API_KEY` dans `.env`.

### 1. Démarrer une session

```bash
curl -X POST http://localhost:8000/api/sessions/start \
  -H "X-API-Key: $KEY"
```

Réponse :

```json
{
  "session_id": "uuid",
  "status": "active",
  "messages_used": 0,
  "remaining_messages": 20,
  "ttl_seconds": 1800
}
```

### 2. Envoyer un message

```bash
curl -X POST http://localhost:8000/api/sessions/$SID/message \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Où se trouve le service de radiologie ?","language":"fr","input_type":"text"}'
```

Réponse :

```json
{
  "session_id": "uuid",
  "answer": "Le service de radiologie se trouve au premier étage.",
  "messages_used": 1,
  "remaining_messages": 19,
  "ttl_seconds": 1800,
  "status": "active"
}
```

### 3. Consulter l'état

```bash
curl http://localhost:8000/api/sessions/$SID/status -H "X-API-Key: $KEY"
```

### 4. Terminer la session

```bash
curl -X POST http://localhost:8000/api/sessions/$SID/end -H "X-API-Key: $KEY"
```

Réponse :

```json
{
  "session_id": "uuid",
  "status": "closed",
  "message": "La session est terminée."
}
```

### 5. Tester l'expiration

Attendre 30 minutes (ou réduire `SESSION_TTL_SECONDS` dans `.env`), puis :

```bash
curl http://localhost:8000/api/sessions/$SID/status -H "X-API-Key: $KEY"
# -> 404 Session inexistante ou expirée.
```

## Tests pytest

```bash
pytest -v
```

Les tests utilisent un faux Redis en mémoire ; aucune dépendance externe requise.

## Architecture

```
app/
├── main.py              # FastAPI app + lifespan
├── config.py            # Settings Pydantic v2 (.env)
├── schemas.py           # Modèles de requêtes/réponses
├── dependencies.py      # Redis + API key
├── services/
│   └── session_service.py  # Logique métier + pipeline RAG fictif
└── routes/
    └── sessions.py      # 4 endpoints
tests/
└── test_sessions.py     # 12 tests pytest-asyncio
```

## Intégration future du pipeline IA

La fonction `process_with_rag()` dans `app/services/session_service.py` est le point
d'entrée prévu pour brancher le vrai pipeline TONTOUMA-BOT :

- transcription audio wolof ;
- traduction Wolof → Français ;
- détection d'intention ;
- recherche hybride BM25 + embeddings ;
- MMR ;
- reranking ;
- génération avec Qwen via Groq ;
- traduction Français → Wolof ;
- synthèse vocale Wolof.

Signature stable :

```python
async def process_with_rag(text, language, history) -> dict
```

Retourne `answer`, `language`, `sources`, `confidence`.

## Améliorations possibles pour la production

- Rate limiting par IP (slowapi ou Redis token bucket).
- Persistence Redis activée (AOF / RDB).
- Logs structurés JSON (structlog) sans contenu d'historique.
- Authentification JWT au lieu d'une clé API statique.
- Monitoring Prometheus + Grafana.
- CI/CD GitHub Actions (lint, tests, build Docker).
- Frontend (React/Vue) avec compte à rebours et compteur de questions.

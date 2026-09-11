# TontumaBot V3 — Guide de déploiement

> Ce guide explique comment déployer l'application pour que l'équipe de développement puisse intégrer les endpoints.

---

## 1. Vue d'ensemble

L'application est exposée sous forme d'API REST + WebSockets via **FastAPI**.  
L'équipe front/mobile n'a qu'à appeler les endpoints HTTP.

```
┌──────────────────────────────────────────┐
│  Front-end / Mobile (équipe dev)         │
│  - appelle /ask, /ask/audio, /health     │
└──────────────┬───────────────────────────┘
               │ HTTPS / WebSocket
               ▼
┌──────────────────────────────────────────┐
│  Serveur TontumaBot V3 déployé           │
│  - FastAPI (uvicorn)                     │
│  - Modèles NLLB, STT, TTS, RAG           │
│  - Documents ingérés depuis HF ou locaux │
└──────────────────────────────────────────┘
```

---

## 2. Prérequis

- Un compte **Hugging Face** (optionnel, pour dataset de documents).
- Une clé **Groq API** obligatoire si `LLM_PROVIDER=groq`.
- (Optionnel) Une clé **Gemini API**.

---

## 3. Créer le dataset de documents sur Hugging Face

### 3.1 Format attendu

Le dataset doit contenir au minimum une colonne `text` avec le contenu du document.

| Colonne | Description |
|---------|-------------|
| `text`  | Contenu textuel du document (obligatoire) |
| `title` | Titre du document (optionnel) |
| `source`| Source du document (optionnel) |

Exemple avec `datasets` Python :

```python
from datasets import Dataset

documents = [
    {
        "text": "Pour obtenir un extrait de naissance...",
        "title": "Procédure naissance",
        "source": "guide_etat_civil.txt",
    },
    {
        "text": "Le guichet CMU se trouve au bâtiment D...",
        "title": "Orientation CMU",
        "source": "orientation_hopital.txt",
    },
]

ds = Dataset.from_list(documents)
ds.push_to_hub("ton-username/tontumabot-documents", private=True)
```

### 3.2 Variables d'environnement

Dans `.env` du serveur déployé :

```bash
HF_DOCUMENTS_DATASET=ton-username/tontumabot-documents
HF_DATASET_SPLIT=train
HF_TEXT_COLUMN=text
HF_TITLE_COLUMN=title
HF_SOURCE_COLUMN=source
```

---

## 4. Méthode 1 : Docker (recommandée)

### 4.1 Construire l'image

```bash
cd V3
docker build -t tontumabot-v3 .
```

### 4.2 Lancer le conteneur

```bash
docker run -d \
  -p 8008:8008 \
  --env-file .env \
  -v tontumabot_cache:/app/data/hf_cache \
  --name tontumabot \
  tontumabot-v3
```

### 4.3 Vérifier

```bash
curl http://localhost:8008/health
```

---

## 5. Méthode 2 : Hugging Face Spaces

### 5.1 Créer un Space

1. Allez sur [https://huggingface.co/spaces](https://huggingface.co/spaces).
2. Créez un Space **Docker** (blank).
3. Poussez `Dockerfile`, `requirements.txt`, `app.py`, `src/`, etc.

### 5.2 Variables secrètes

Dans l'onglet **Settings → Secrets**, ajoutez :

- `GROQ_API_KEY`
- `HF_DOCUMENTS_DATASET`
- etc.

### 5.3 Limites

Le tier gratuit a 16 Go RAM et 2 vCPU.  
Les modèles NLLB + Oolel peuvent être lourds. Privilégiez un Space payant ou un VPS.

---

## 6. Méthode 3 : Render / Railway / Fly.io

Ces services supportent Docker.

### Render
1. Créez un Web Service.
2. Connectez le repo Git.
3. Définissez `Dockerfile` comme build command.
4. Ajoutez les variables d'environnement.

### Railway
1. Créez un projet.
2. Déployez depuis GitHub.
3. Ajoutez les variables dans Variables.

### Fly.io
```bash
fly launch
cp .env.example .env
fly secrets set GROQ_API_KEY=xxx HF_DOCUMENTS_DATASET=...
fly deploy
```

---

## 7. Endpoints à communiquer à l'équipe dev

| Endpoint | Méthode | Usage |
|----------|---------|-------|
| `/health` | GET | Vérifier la santé |
| `/models` | GET | Vérifier les modèles |
| `/ask` | POST | Question texte |
| `/ask/audio` | POST | Question audio |
| `/translate` | GET | Test traduction |
| `/admin/documents` | POST/GET | Gestion documents |
| `/ws/ask` | WebSocket | Streaming |

Exemple d'intégration front :

```javascript
const res = await fetch('https://ton-domaine.com/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'Comment obtenir un extrait de naissance ?',
    language: 'fr',
    tts: false,
  }),
});
const data = await res.json();
console.log(data.response_fr);
```

---

## 8. Checklist avant mise en production

- [ ] `.env` configuré avec `GROQ_API_KEY` (ou `GEMINI_API_KEY` / modèle local).
- [ ] `HF_DOCUMENTS_DATASET` pointe vers un dataset privé et valide.
- [ ] Les modèles sont téléchargés en cache (`~/.cache/huggingface` ou dans le conteneur).
- [ ] L'URL publique est accessible.
- [ ] HTTPS est activé.
- [ ] Les endpoints `/health` et `/models` répondent 200.
- [ ] Un test `/ask` retourne une réponse cohérente.

---

## 9. Dépannage

| Problème | Solution |
|----------|----------|
| Container échoue au démarrage | Vérifier `.env` et logs : `docker logs tontumabot` |
| Lenteur excessive | Passer sur une machine avec plus de CPU/RAM ou GPU |
| Modèle Oolel ne charge pas | Vérifier `OOLEL_TTS_REPO` et mémoire disponible |
| Documents non ingérés | Vérifier `HF_DOCUMENTS_DATASET` et le format du dataset |
| Erreur 500 sur `/ask` | Vérifier `GROQ_API_KEY` et connexion Internet |

---

## 10. Ressources

- API docs : http://localhost:8008/docs
- Notebook démo : `notebooks/demo_TontumaBot_V3.ipynb`
- Fichier coach : `coach.md`

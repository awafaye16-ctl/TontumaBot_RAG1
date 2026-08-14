# TontumaBot — Assistant administratif RAG multilingue

Assistant intelligent **multilingue** (wolof / français), **multimodal** (texte, audio, image) et **orienté RAG** pour les procédures administratives sénégalaises.

L'utilisateur pose une question en **wolof ou en français**, par **texte, voix ou photo** ; le système la normalise en français, recherche l'information dans une **base documentaire vectorielle** (ou une base de démonstration), génère une réponse avec un LLM, puis la restitue **dans la langue et le format d'origine** (texte ou voix).

Les documents de la base sont gérés via l'**interface admin** (`/admin`) : upload de fichiers (TXT, MD, PDF) ou collage de texte, découpage en chunks, indexation d'embeddings et suppression.

La saisie est **multimodale** : texte, **enregistrement vocal direct via le micro** (🎙️ dans l'interface), photo/document (OCR) ou réponse vocale wolof (TTS).

## Architecture

```
Entrée (texte / audio / image)
   │  ├─ Texte brut ──────────────┐
   │  ├─ Audio wolof → Speech-to-Text │
   │  └─ Image/doc → OCR          │
   ▼                               ▼
          Texte unifié (WO / FR)
                    │
              Détection de langue
              ↙                ↘
          Wolof              Français
            │                   │
     NLLB-200 WO→FR            │
            ↘                   ↓
              Requête en FRANÇAIS
                    │
              Router d'intention
            ↙                ↘
      procédure           orientation
            │                   │
   Recherche hybride    Recherche filtrée
   (BM25 + vectoriel)   (lieu/service/coût)
            ↘                ↙
            Contexte FR
                 │
            LLM générateur (Groq / Gemini)
                 │
       Réponse en français
          ↙            ↘
     Français        Wolof
        │              │
  Texte réponse   NLLB-200 FR→WO
        │              │
        └─────┬────────┘
        Texte réponse (langue d'origine)
              │
        (option) TTS → note vocale
```

### Pourquoi traduire en français ?

Au lieu d'un pipeline wolof séparé, l'architecture mutualise **une seule base documentaire, un seul moteur de recherche, un seul LLM**. Le wolof est une *langue d'interface* ; le français est la *langue intermédiaire de traitement*.

## V1 — Structure

```
V1/
├── app.py                      # API FastAPI (chat + admin RAG)
├── demo.py                     # démo CLI sans modèles lourds
├── requirements.txt
├── .env.example                # modèle du fichier .env (clés API)
├── Dockerfile                  # image de production (torch CPU)
├── docker-compose.yml          # orchestration (ports, volumes, healthcheck)
├── .dockerignore
├── data/
│   ├── seed_docs.py            # base documentaire de démonstration
│   ├── documents/              # documents à ingérer (TXT/MD/PDF)
│   ├── speaker_embedding.npy   # voix TTS (xvector CMU Arctic)
│   └── chroma/                 # base vectorielle persistante (auto)
├── static/
│   ├── index.html              # interface web de test (mode chat + micro)
│   └── admin.html              # interface admin RAG (upload docs)
└── src/
    ├── config.py               # configuration depuis .env
    ├── pipeline.py             # orchestration du flux complet
    ├── vectorstore.py          # service ChromaDB + embeddings
    ├── ingestion.py            # extraction + chunking des documents
    ├── input/
    │   ├── text.py             # entrée texte
    │   ├── ocr.py              # pytesseract (image/document)
    │   └── stt.py              # whisper LoRA wolof (audio)
    ├── language/detector.py    # détection wolof/français
    ├── translation/nllb.py     # NLLB-200 WO<->FR
    ├── intent/router.py        # procédure / orientation
    ├── retrieval/
    │   ├── hybrid.py           # BM25 + vectoriel
    │   └── filtered.py         # recherche filtrée
    ├── generation/llm.py       # Groq / Gemini (fallback sans clé)
    └── tts/tts.py              # TTS wolof SpeechT5 (+ fallback edge-tts)
```

## Installation

### Prérequis

- **Python 3.10+**
- **Tesseract OCR** (pour l'OCR image)

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-fra
```

### Démarrage rapide

```bash
cd V1

# 1. Environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 2. Dépendances
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# → renseigner GROQ_API_KEY ou GEMINI_API_KEY dans .env

# 4. Lancement
PORT=8001 python app.py
```

Ouvrir <http://localhost:8001> (interface web de test). La documentation Swagger de l'API est sur <http://localhost:8001/docs>.

> Le port 8000 est utilisé par défaut ; si occupé, utilisez un autre port via `PORT=...`.

## Docker

### Démarrage avec Docker

```bash
cd V1

# 1. Configuration
cp .env.example .env
# → renseigner GROQ_API_KEY ou GEMINI_API_KEY dans .env

# 2. Construire et lancer
docker compose up --build
```

- Interface : <http://localhost:8001> (port hôte `8001` → conteneur `8000`).
- Le **premier build est long** (torch CPU, transformers, whisper… ~8 Go de téléchargement).
- **Volumes persistants** :
  - `./data:/app/data` — base ChromaDB + voix TTS (`speaker_embedding.npy`) ;
  - `./uploads:/app/uploads` — fichiers ingérés / OCR / STT ;
  - `./models:/app/models` — caches HuggingFace (NLLB, embeddings, whisper, TTS) téléchargés au premier usage et conservés entre les redémarrages.

### Notes Docker

- **Torch CPU uniquement** : installé depuis l'index CPU (`pip install torch --index-url .../whl/cpu`), l'image est ~10× plus petite que la variante CUDA. Les modèles tournent sur CPU (plus lent, mais zéro GPU requis).
- **Modèles téléchargés au premier usage** : le conteneur démarre vite ; NLLB, embeddings, whisper et le TTS wolof se téléchargent dans `models/` à la première requête qui en a besoin.
- **Arrêt** : `docker compose down` (les volumes `./data`, `./uploads`, `./models` restent sur l'hôte).
- **Sans clé API LLM**, le générateur retombe en *mode fallback* : il renvoie le passage le plus pertinent du contexte retrouvé.

## Configuration (.env)

| Variable | Description | Défaut |
|---|---|---|
| `GROQ_API_KEY` | Clé API Groq (LLM) | vide |
| `GEMINI_API_KEY` | Clé API Google Gemini (LLM) | vide |
| `LLM_PROVIDER` | `groq` ou `gemini` | `groq` |
| `NLLB_MODEL` | Modèle de traduction | `bilalfaye/nllb-200-distilled-600M-wo-fr-en` |
| `EMBED_MODEL` | Modèle d'embedding (multilingue FR/WO) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `WOLOF_TTS_MODEL` | Modèle TTS wolof (SpeechT5) — vide = voix FR edge-tts | `bilalfaye/speecht5_tts-wolof-v0.2` |
| `STT_MODEL_PATH` | Chemin du LoRA whisper wolof (sinon Whisper small) | `./wolof-whisper-small-lora` |
| `STT_LANGUAGE` | Langue STT (`wo`) ou vide = auto | vide |
| `OCR_LANGS` | Langues tesseract | `fra+eng` |
| `HOST` / `PORT` | Hôte / port serveur | `0.0.0.0` / `8000` |

**Sans clé API LLM**, le générateur retombe en *mode fallback* : il renvoie le passage le plus pertinent du contexte retrouvé.

## API REST

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interface web (chat) |
| `GET` | `/admin` | Interface admin RAG (upload de documents) |
| `GET` | `/health` | État du serveur + config LLM + nb documents indexés |
| `POST` | `/ask` | Question texte (FR/WO) — body `{question, provider?, tts?}` |
| `POST` | `/ask/audio` | Note vocale (multipart `file`) → STT → pipeline |
| `POST` | `/ask/image` | Photo/doc (multipart `file`) → OCR → pipeline |
| `GET` | `/translate` | Test traduction NLLB (`text`, `direction=wo2fr\|fr2wo`) |
| `POST` | `/admin/documents` | Ingérer un document (multipart `file` TXT/MD/PDF, ou `text` brut) |
| `GET` | `/admin/documents` | Lister les documents indexés |
| `DELETE` | `/admin/documents/{id}` | Supprimer un document (et ses chunks) |
| `POST` | `/admin/documents/clear` | Vider toute la base vectorielle |
| `GET` | `/response.mp3` | Dernière réponse vocale générée |

### Exemples

```bash
# Question texte wolof
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"dama beug wout kayitu juddu ?"}'

# Question texte + réponse vocale
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Où dois-je déposer ma demande ?","tts":true}'

# Image (OCR)
curl -X POST http://localhost:8001/ask/image -F "file=@capture.png"

# Ingérer un document (admin RAG)
curl -X POST http://localhost:8001/admin/documents \
  -F "file=@data/documents/cni.md" -F "title=Procédure carte d'identité"

# Lister les documents indexés
curl http://localhost:8001/admin/documents

# Traduction NLLB directe
curl "http://localhost:8001/translate?text=dama%20soxla%20doctoor&direction=wo2fr"
```

### Réponse type

```json
{
  "trace": {
    "input_lang": "wo",
    "wolof_to_french": {"query": "Je veux un certificat de naissance.", "seconds": 6.4},
    "intent": "procedure",
    "search": "hybrid",
    "context": "…",
    "french_to_wolof": {"seconds": 6.2}
  },
  "response_fr": "Pour obtenir un extrait de naissance…",
  "response_wo": "Soo bëggee am kayitu juddu gi…",
  "response": "Soo bëggee am kayitu juddu gi…",
  "audio": "…/response.mp3"
}
```

## Pipeline détaillé

1. **Entrée** : texte brut, **enregistrement micro** (MediaRecorder → `/ask/audio`), ou image (pytesseract OCR).
2. **Unification** : toutes les entrées deviennent du texte.
3. **Détection de langue** : lexique wolof → `wo` ou `fr` (`src/language/detector.py`).
4. **Traduction WO→FR** : NLLB-200 fine-tuné wolof (`bilalfaye/nllb-200-distilled-600M-wo-fr-en`).
5. **Router d'intention** : mots-clés d'orientation (lieu/service/coût) → `orientation`, sinon `procedure`.
6. **Recherche** (la base vectorielle ChromaDB est utilisée si elle contient des documents, sinon la base seed de démo) :
   - *procédure* → **hybride** : **BM25** (lexical) + **vectoriel** (embeddings `paraphrase-multilingual-MiniLM-L12-v2`).
   - *orientation* → **recherche vectorielle** sur les chunks, ou **recherche filtrée** seed par métadonnées (`lieu`, `service`, `cout`).
7. **Génération** : LLM Groq (llama-3.3-70b) ou Gemini, strictement sur le contexte retrouvé.
8. **Retour à la langue d'origine** : si la question était en wolof, la réponse FR est re-traduite en wolof.
9. **Sortie** : texte (+ **TTS** wolof SpeechT5 optionnel pour la note vocale, fallback edge-tts FR). Le texte est nettoyé avant synthèse (chiffres → mots, ponctuation douce, mots anglais retirés) et découpé en phrases courtes, car le modèle SpeechT5 wolof dégénère en bruit sur les textes longs ou les chiffres.

## Composants modèles

| Composant | Modèle / outil | Rôle |
|---|---|---|
| Traduction | `bilalfaye/nllb-200-distilled-600M-wo-fr-en` | Wolof ↔ français |
| Speech-to-Text | `./wolof-whisper-small-lora` (LoRA) ou Whisper small | Audio wolof → texte |
| OCR | `pytesseract` + tesseract | Image / document → texte |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Indexation vectorielle |
| Stockage vectoriel | `chromadb` (persistant) | BD vectorielle des chunks |
| Recherche | `rank-bm25` + ChromaDB | Contexte RAG (hybride) |
| Extraction docs | `pypdf` | PDF → texte (ingestion) |
| LLM | Groq (`llama-3.3-70b-versatile`) / Gemini | Génération |
| TTS | `bilalfaye/speecht5_tts-wolof-v0.2` (SpeechT5 + HiFi-GAN) | Texte wolof/français → audio |
| Voix TTS | xvector CMU Arctic (voix `slt`) | Identité vocale du TTS |
| Déploiement | Docker (torch CPU) | Conteneurisation + persistence volumes |

## Tests rapides

```bash
cd V1 && source .venv/bin/activate

# Suite de tests (rapide, sans modèles lourds)
python -m pytest

# Inclure la traduction NLLB réelle (charge ~600M, ~15s)
RUN_NLLB_TESTS=1 python -m pytest tests/test_nllb.py

# Sans serveur (détection + intention + recherche)
python demo.py

# Traduction NLLB réelle
python src/translation/nllb.py
```

### Couverture des tests (`tests/`)

| Fichier | Cible |
|---|---|
| `test_language.py` | Détection wolof/français (phrases types, cas vides) |
| `test_intent.py` | Router procédure/orientation (lieu, service, coût, faux positifs) |
| `test_retrieval.py` | BM25 hybride + recherche filtrée par métadonnées |
| `test_pipeline.py` | Flux complet en mode fallback (sans API) |
| `test_api.py` | Endpoints FastAPI via TestClient (`/health`, `/ask`, `/`) |
| `test_nllb.py` | Traduction NLLB réelle (désactivé sans `RUN_NLLB_TESTS=1`) |
| `test_tts.py` | TTS : fallback edge-tts sans modèle wolof, moteur actif |

Résultat actuel : **29 passés, 3 désactivés (NLLB), 0 échec** (32 passés avec `RUN_NLLB_TESTS=1`).

## Admin RAG (base documentaire vectorielle)

L'interface **`/admin`** permet de gérer la base qui alimente le chat :

- **Upload de fichiers** : TXT, MD, PDF (glisser-déposer ou clic).
- **Collage de texte** : collez directement une procédure, un arrêté, un guide…
- Le texte est **découpé en chunks** (~600 caractères, chevauchement 80) puis **encodé** (embeddings multilingues) et indexé dans **ChromaDB** (`data/chroma/`, persistant).
- **Liste des documents** : nombre de chunks, date d'ajout, suppression individuelle ou globale.

Lorsqu'une question est posée dans le chat, la recherche hybride **BM25 + vectorielle** interroge cette base. Si elle est vide, le système retombe sur la base de démonstration `data/seed_docs.py`.

### Flux d'utilisation recommandé

1. Ouvrir `http://localhost:8001/admin` et ingérer vos documents.
2. Ouvrir `http://localhost:8001/` et poser une question (FR ou WO).
3. La réponse s'appuie alors uniquement sur votre base documentaire.

## Roadmap V2 (idées)

- Améliorer la qualité TTS wolof (essayer `galsenai/xTTS-v2-wolof` ou API Khaya AI).
- Voix TTS wolof native additionnelles (variation de la voix `slt`).
- Authentification API et déploiement (Streamlit/Frontend dédié).
- Re-embedding automatique et gestion des doublons.
- Support GPU optionnel dans Docker (`--gpus all` + index CUDA).
# TontumaBot V3 — Documentation API

> Assistant administratif sénégalais multilingue (Wolof / Français)  
> RAG hybride · Reranker cross-encoder · TTS Oolel-Voices · Streaming WebSocket

**Base URL** : `http://localhost:8008`  
**Version** : 3.0.0  
**Python** : 3.11+

---

## Table des matières

- [TontumaBot V3 — Documentation API](#tontumabot-v3--documentation-api)
  - [Table des matières](#table-des-matières)
  - [Pipeline](#pipeline)
  - [Authentification](#authentification)
  - [Endpoints REST](#endpoints-rest)
    - [`GET /health`](#get-health)
    - [`GET /models`](#get-models)
    - [`POST /ask`](#post-ask)
    - [`POST /ask/audio`](#post-askaudio)
    - [`GET /translate`](#get-translate)
    - [`GET /response.wav`](#get-responsewav)
    - [`POST /admin/documents`](#post-admindocuments)
    - [`GET /admin/documents`](#get-admindocuments)
    - [`DELETE /admin/documents/{document_id}`](#delete-admindocumentsdocument_id)
    - [`POST /admin/documents/clear`](#post-admindocumentsclear)
    - [`POST /eval/ragas`](#post-evalragas)
  - [WebSockets](#websockets)
    - [`WS /ws/ask`](#ws-wsask)
    - [`WS /ws/stt`](#ws-wsstt)
  - [Pipeline Trace](#pipeline-trace)
  - [Exemples curl](#exemples-curl)
    - [Question texte français](#question-texte-français)
    - [Question texte wolof avec TTS](#question-texte-wolof-avec-tts)
    - [Question audio (micro WebM)](#question-audio-micro-webm)
    - [Ajouter un document PDF](#ajouter-un-document-pdf)
    - [Ajouter du texte brut](#ajouter-du-texte-brut)
    - [Lister les documents](#lister-les-documents)
    - [Supprimer un document](#supprimer-un-document)
    - [Vider la base](#vider-la-base)
    - [Traduction rapide WO→FR](#traduction-rapide-wofr)
    - [Traduction rapide FR→WO](#traduction-rapide-frwo)
    - [Santé du serveur](#santé-du-serveur)
  - [Configuration .env](#configuration-env)
  - [Dépendances principales](#dépendances-principales)

---

## Pipeline

```
Entrée utilisateur
  │
  ├─ 🎙️  Audio WebM/WAV  →  [STT Whisper wolof local]        →  texte
  ├─ 🖼️  Image/PDF       →  [Tesseract OCR]                  →  texte
  └─ ⌨️  Texte brut      →  (direct)                         →  texte
  │
  ▼
[Détecteur lexical wolof/français]
  │
  ├─ wolof → [NLLB bilalfaye/600M  WO→FR]  →  question en français
  └─ français → (direct)                   →  question en français
  │
  ▼
[Router d'intention]  →  procedure | orientation
  │
  ▼
RAG Pipeline (ChromaDB peuplée, sinon seed en mémoire)
  ├─ 1. Hybrid search  BM25 + vectoriel          fetch_k = 20 candidats
  ├─ 2. MMR  diversité sémantique                →  8 chunks
  └─ 3. Reranker  cross-encoder ms-marco         →  top 3 chunks (≤ 800 chars)
  │
  ▼
[LLM qwen/qwen3.8-27b via Groq]  (ou Gemini gemini-2.0-flash)
  →  réponse en français (max 300 tokens)
  │
  ├─ si entrée wolof → [NLLB Lahad/600M  FR→WO]  →  réponse en wolof
  └─ si entrée français → (direct)
  │
  ▼
[TTS optionnel]
  ├─ Tentative 1 : Oolel-Voices  (voice cloning voix_clone.mp3)
  ├─ Tentative 2 : SpeechT5 bilalfaye wolof
  └─ Tentative 3 : edge-tts  fr-FR-DeniseNeural
  │
  ▼
Réponse JSON  { response, response_fr, response_wo?, audio? }
```

---

## Authentification

Aucune pour les endpoints publics (`/ask`, `/ask/audio`, `/translate`).  
Les endpoints `/admin/*` et `/eval/*` doivent être **protégés en production**
(reverse proxy avec authentification basique ou token Bearer).

---

## Endpoints REST

---

### `GET /health`

État du serveur et de tous ses composants.

**Réponse** `200 OK`

```json
{
  "status": "ok",
  "version": "3.0.0",
  "llm_provider": "groq",
  "llm_ready": true,
  "tts_engine": "oolel",
  "nllb_wo_fr": "/chemin/vers/Wo_fr_bilalfaye...",
  "nllb_fr_wo": "/chemin/vers/Fr-Wo_Lahad...",
  "stt_model":  "/chemin/vers/stt_wolof-whisper-small-lora",
  "reranker":   "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "n_documents": 2,
  "n_chunks": 41
}
```

| Champ          | Type   | Description                                 |
| -------------- | ------ | ------------------------------------------- |
| `status`       | string | `ok` si le serveur répond                   |
| `version`      | string | Version de l'API                            |
| `llm_provider` | string | `groq` \| `gemini` \| `local`               |
| `llm_ready`    | bool   | `true` si la clé API est configurée         |
| `tts_engine`   | string | `oolel` \| `speecht5` \| `edge`             |
| `nllb_wo_fr`   | string | Chemin/ID du modèle NLLB WO→FR              |
| `nllb_fr_wo`   | string | Chemin/ID du modèle NLLB FR→WO              |
| `stt_model`    | string | Chemin/ID du modèle Whisper wolof           |
| `reranker`     | string | Modèle cross-encoder                        |
| `n_documents`  | int    | Nombre de documents distincts dans ChromaDB |
| `n_chunks`     | int    | Nombre total de chunks vectorisés           |

---

### `GET /models`

Liste l'ensemble des modèles et embeddings configurés, avec leur statut (local / Hub, présent ou non, poids disponibles). Utile pour le développement et le débogage.

**Réponse** `200 OK`

```json
{
  "status": "ok",
  "models": {
    "llm": {
      "provider":    "groq",
      "local_model": "Qwen/Qwen2.5-7B-Instruct",
      "quant":       "4bit",
      "ready":       true
    },
    "nllb_wo_fr": {
      "task":   "wolof → français",
      "source": "C:/.../Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en",
      "status": {
        "type":    "local",
        "path":    "C:/...",
        "exists":  true,
        "weights": true
      }
    },
    "nllb_fr_wo": {
      "task":   "français → wolof",
      "source": "C:/.../Fr-Wo_Lahadnllb200-francais-wolof",
      "status": { "type": "local", "path": "C:/...", "exists": true, "weights": true }
    },
    "stt": {
      "task":   "wolof whisper",
      "source": "C:/.../stt_wolof-whisper-small-lora",
      "status": { "type": "local", "path": "C:/...", "exists": true, "weights": true }
    },
    "embeddings": {
      "task":   "sentence embeddings",
      "source": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    },
    "reranker": {
      "task":   "cross-encoder reranking",
      "source": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "tts": {
      "engine":      "oolel",
      "oolel_repo":  "soynade-research/Oolel-Voices",
      "speecht5_fb": "bilalfaye/speecht5_tts-wolof-v0.2"
    }
  }
}
```

**Exemple curl**

```bash
curl http://localhost:8008/models
```

---

### `POST /ask`

Question texte (wolof ou français). Pipeline complet : détection langue → RAG → LLM → TTS optionnel.

**Content-Type** : `application/json`

**Body**

```json
{
  "question":   "Quels documents pour une carte d'identité ?",
  "provider":   "groq",
  "tts":        false,
  "tts_engine": "oolel"
}
```

| Champ        | Type   | Requis  | Défaut        | Description                     |
| ------------ | ------ | ------- | ------------- | ------------------------------- |
| `question`   | string | **oui** | —             | Question en wolof ou français   |
| `provider`   | string | non     | depuis `.env` | `groq` \| `gemini` \| `local`   |
| `tts`        | bool   | non     | `false`       | Générer l'audio de la réponse   |
| `tts_engine` | string | non     | depuis `.env` | `oolel` \| `speecht5` \| `edge` |

**Réponse** `200 OK`

```json
{
  "trace": {
    "input_lang": "wo",
    "wolof_to_french": {
      "model": "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
      "result": "Comment obtenir un certificat de naissance ?",
      "latency_ms": 1230
    },
    "intent": "procedure",
    "retrieval": {
      "source": "chromadb",
      "n_candidates": 20,
      "n_mmr": 8,
      "n_reranked": 3,
      "latency_total_ms": 412,
      "reranker_scores": [7.06, -0.91, -10.75],
      "chunks": [
        { "rank": 1, "score": 7.06, "text": "Pour obtenir une CNI..." }
      ]
    },
    "context": "Pour obtenir une CNI...",
    "llm": { "provider": "groq", "latency_ms": 980 },
    "french_to_wolof": {
      "model": "Lahad/nllb200-francais-wolof",
      "latency_ms": 1100
    },
    "tts": { "engine": "oolel-voices", "success": true, "latency_ms": 8200 },
    "total_latency_ms": 12340
  },
  "response_fr": "Pour obtenir une carte d'identité nationale...",
  "response_wo": "Soo bëggee am kayitu juddu gi...",
  "response":    "Soo bëggee am kayitu juddu gi...",
  "audio":       "/static/response.wav"
}
```

| Champ         | Type   | Description                                                      |
| ------------- | ------ | ---------------------------------------------------------------- |
| `trace`       | object | Détails du pipeline (voir [Pipeline Trace](#pipeline-trace))     |
| `response_fr` | string | Réponse toujours en français                                     |
| `response`    | string | Réponse dans la langue de l'utilisateur                          |
| `response_wo` | string | Réponse en wolof (uniquement si `input_lang == "wo"`)            |
| `audio`       | string | URL `/static/response.wav` (uniquement si `tts: true` et succès) |
| `session_id`  | string | Identifiant de session (généré si absent de la requête)          |
| `session`     | object | `{statut, questions_restantes, session_fermee, message?}` — présent uniquement si `session_id` a été fourni. `message` apparaît quand `session_fermee: true` |

**Erreurs**

| Code  | Cause                                 |
| ----- | -------------------------------------- |
| `400` | Question vide                         |
| `409` | Limite de questions de la session atteinte — démarrer une nouvelle session |
| `410` | Session expirée par inactivité — démarrer une nouvelle session |
| `500` | Erreur interne LLM, TTS ou traduction |

---

### `POST /ask/audio`

Envoie un fichier audio (WebM, WAV, MP3, M4A) pour transcription STT puis pipeline RAG complet.

**Content-Type** : `multipart/form-data`

**Body**

| Champ        | Type   | Requis  | Description                                  |
| ------------ | ------ | ------- | -------------------------------------------- |
| `file`       | file   | **oui** | Fichier audio (webm, wav, mp3, m4a…)         |
| `tts`        | bool   | non     | Générer l'audio en réponse (défaut: `false`) |
| `tts_engine` | string | non     | `oolel` \| `speecht5` \| `edge`              |
| `provider`   | string | non     | `groq` \| `gemini` \| `local`                |

**Réponse** : Identique à `POST /ask`, avec en plus `trace.stt_text` = transcription.

**Erreurs**

| Code  | Cause                                            |
| ----- | ------------------------------------------------ |
| `400` | Aucun texte transcrit (audio vide ou silencieux) |
| `413` | Fichier > 50 Mo                                  |
| `500` | Échec décodage audio ou STT                      |

---

### `GET /translate`

Test rapide de traduction NLLB (sans RAG ni LLM).

**Paramètres query**

| Param       | Type   | Défaut                         | Description                        |
| ----------- | ------ | ------------------------------ | ---------------------------------- |
| `text`      | string | `"Jërejëf lool ci dimbal bi."` | Texte à traduire                   |
| `direction` | string | `wo2fr`                        | `wo2fr` (WO→FR) ou `fr2wo` (FR→WO) |

**Réponse** `200 OK`

```json
{
  "input":     "Jërejëf lool ci dimbal bi.",
  "output":    "Merci beaucoup pour l'aide.",
  "direction": "wo→fr",
  "model":     "/chemin/vers/Wo_fr_bilalfaye...",
  "seconds":   1.23
}
```

---

### `GET /response.wav`

Retourne le dernier fichier audio TTS généré.

**Réponse** : Fichier audio WAV (`audio/wav`)

**Erreurs**

| Code  | Cause                                  |
| ----- | -------------------------------------- |
| `404` | Aucun audio généré depuis le démarrage |

---

### `POST /admin/documents`

Ingère un document dans ChromaDB. Découpage automatique en chunks (~600 chars, overlap 80), vectorisation et indexation.

**Content-Type** : `multipart/form-data`

**Body** — fournir `file` **OU** `text`

| Champ   | Type   | Requis | Description                      |
| ------- | ------ | ------ | -------------------------------- |
| `file`  | file   | non*   | Fichier `.txt`, `.md` ou `.pdf`  |
| `text`  | string | non*   | Texte brut à indexer directement |
| `title` | string | non    | Titre affiché dans la liste      |

*\*Au moins l'un des deux est requis.*

**Réponse** `200 OK`

```json
{
  "ok": true,
  "chunks": 7,
  "title": "Procédure passeport biométrique"
}
```

**Erreurs**

| Code  | Cause                                                                             |
| ----- | --------------------------------------------------------------------------------- |
| `400` | Aucun fichier/texte fourni, format non supporté (seuls txt/md/pdf), ou texte vide |
| `500` | Erreur d'extraction (PDF corrompu…)                                               |

---

### `GET /admin/documents`

Liste tous les documents indexés dans ChromaDB (dédoublonnés par `document_id`).

**Réponse** `200 OK`

```json
{
  "documents": [
    {
      "id":     "1720000000000",
      "title":  "Procédure CNI",
      "chunks": 7,
      "added":  "2024-07-03T12:00:00+00:00"
    }
  ],
  "total_chunks": 41
}
```

---

### `DELETE /admin/documents/{document_id}`

Supprime un document et tous ses chunks de ChromaDB.

**Paramètre path** : `document_id` (valeur du champ `id` retourné par `GET /admin/documents`)

**Réponse** `200 OK`

```json
{
  "ok": true,
  "deleted_chunks": 7
}
```

**Erreurs**

| Code  | Cause                     |
| ----- | ------------------------- |
| `404` | `document_id` introuvable |

---

### `POST /admin/documents/clear`

Supprime **tous** les documents de ChromaDB.

**Réponse** `200 OK`

```json
{
  "ok": true,
  "deleted_chunks": 41
}
```

---

### `POST /eval/ragas`

Lance une évaluation de qualité RAG sur un jeu de test (métriques RAGAS : faithfulness, answer relevancy, context precision…).

**Content-Type** : `application/json`

**Body**

```json
{
  "provider": "groq",
  "test_cases": [
    {
      "question":         "Quels documents pour une CNI ?",
      "reference_answer": "Extrait de naissance, deux photos d'identité et reçu de paiement.",
      "category":         "procedure",
      "language":         "fr"
    }
  ]
}
```

**Réponse** `200 OK`

```json
{
  "summary": {
    "total": 1,
    "avg_faithfulness": 0.91,
    "avg_answer_relevancy": 0.87
  },
  "output_file": "/chemin/vers/resultats_ragas/evaluation_v3.json"
}
```

---

## WebSockets

### `WS /ws/ask`

Streaming LLM token par token. Recommandé pour l'interface chat (réponse affichée progressivement).

**Protocole client → serveur** (JSON, un seul message)

```json
{
  "question":   "Quels documents pour un passeport ?",
  "provider":   "groq",
  "tts":        false,
  "tts_engine": "oolel"
}
```

**Protocole serveur → client** (JSON, plusieurs messages)

| Type            | Contenu                                                      | Description                        |
| --------------- | ------------------------------------------------------------ | ---------------------------------- |
| `lang`          | `{"lang": "fr"}`                                             | Langue détectée                    |
| `query_fr`      | `{"text": "..."}`                                            | Question traduite WO→FR (si wolof) |
| `context_ready` | `{"n_chunks": 3}`                                            | RAG terminé                        |
| `token`         | `{"text": "..."}`                                            | Fragment de réponse LLM            |
| `done`          | `{"response_fr":"...","response":"...","response_wo":"..."}` | Réponse complète                   |
| `audio_ready`   | `{"audio": "/static/response_xxx.wav"}`                      | Audio TTS prêt (asynchrone)        |
| `error`         | `{"text": "..."}`                                            | Erreur                             |

---

### `WS /ws/stt`

STT live — transcription d'un flux audio en temps réel.

> ⚠️ Pour les navigateurs, préférer `POST /ask/audio` (plus fiable).  
> `/ws/stt` est destiné aux clients qui gèrent le streaming audio manuellement.

**Protocole client → serveur**

| Type  | Contenu          | Description                        |
| ----- | ---------------- | ---------------------------------- |
| bytes | chunk audio brut | WebM/opus du micro (MediaRecorder) |
| text  | `"stop"`         | Fin de l'enregistrement            |

**Protocole serveur → client** (JSON)

| Type    | Description                          |
| ------- | ------------------------------------ |
| `final` | `{"text": "transcription complète"}` |
| `error` | `{"text": "message d'erreur"}`       |

---

## Pipeline Trace

L'objet `trace` retourné par `/ask` et `/ask/audio` :

| Clé                | Type   | Description                                                                            |
| ------------------ | ------ | -------------------------------------------------------------------------------------- |
| `input_lang`       | string | `wo` ou `fr`                                                                           |
| `greeting`         | bool   | `true` si message de salutation (pas de RAG)                                           |
| `wolof_to_french`  | object | `{model, result, latency_ms}` — si entrée wolof                                        |
| `intent`           | string | `procedure` ou `orientation`                                                           |
| `retrieval`        | object | `{source, n_candidates, n_mmr, n_reranked, latency_total_ms, reranker_scores, chunks}` |
| `context`          | string | Contexte envoyé au LLM (≤ 800 chars)                                                   |
| `n_docs_in_db`     | int    | Nombre de chunks dans ChromaDB au moment de la requête                                 |
| `llm`              | object | `{provider, latency_ms}`                                                               |
| `french_to_wolof`  | object | `{model, latency_ms}` — si entrée wolof                                                |
| `tts`              | object | `{engine, success, latency_ms}` — si TTS demandé                                       |
| `stt_text`         | string | Transcription STT (`/ask/audio` uniquement)                                            |
| `total_latency_ms` | int    | Latence totale du pipeline en millisecondes                                            |

---

## Exemples curl

### Question texte français
```bash
curl -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Combien coûte un passeport ?"}'
```

### Question texte wolof avec TTS
```bash
curl -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "dama beug wout kayitu juddu ?", "tts": true, "tts_engine": "oolel"}'
```

### Question audio (micro WebM)
```bash
curl -X POST http://localhost:8008/ask/audio \
  -F "file=@recording.webm" \
  -F "tts=true" \
  -F "tts_engine=oolel"
```

### Ajouter un document PDF
```bash
curl -X POST http://localhost:8008/admin/documents \
  -F "file=@guide_cni.pdf" \
  -F "title=Procédure CNI"
```

### Ajouter du texte brut
```bash
curl -X POST http://localhost:8008/admin/documents \
  -F "text=Pour obtenir une CNI, fournir un extrait de naissance et deux photos." \
  -F "title=CNI résumé"
```

### Lister les documents
```bash
curl http://localhost:8008/admin/documents
```

### Supprimer un document
```bash
curl -X DELETE http://localhost:8008/admin/documents/1720000000000
```

### Vider la base
```bash
curl -X POST http://localhost:8008/admin/documents/clear
```

### Traduction rapide WO→FR
```bash
curl "http://localhost:8008/translate?text=Jërejëf+lool+ci+dimbal+bi.&direction=wo2fr"
```

### Traduction rapide FR→WO
```bash
curl "http://localhost:8008/translate?text=Comment+obtenir+un+passeport&direction=fr2wo"
```

### Santé du serveur
```bash
curl http://localhost:8008/health
```

---

## Configuration .env

| Variable           | Défaut                                                        | Description                                       |
| ------------------ | ------------------------------------------------------------- | ------------------------------------------------- |
| `MAX_QUESTIONS_PER_SESSION` | `2`                                                   | Nombre de questions autorisées par session avant fermeture (409) |
| `SESSION_INACTIVITY_SECONDS` | `900`                                                | Expiration par inactivité, en secondes (alias accepté : `SESSION_TTL_SECONDS`) — au-delà, 410 |
| `MEMORY_MAX_TURNS` | `5`                                                            | Nombre de tours (Q+R) conservés pour la reformulation de suivi |
| `GROQ_API_KEY`     | —                                                             | Clé API Groq (obligatoire si `LLM_PROVIDER=groq`) |
| `GEMINI_API_KEY`   | —                                                             | Clé API Google Gemini                             |
| `LLM_PROVIDER`     | `groq`                                                        | `groq` \| `gemini` \| `local`                     |
| `LOCAL_LLM_MODEL`  | `Qwen/Qwen2.5-7B-Instruct`                                    | Modèle local HuggingFace                          |
| `LOCAL_LLM_QUANT`  | `4bit`                                                        | `4bit` \| `8bit` \| `fp16`                        |
| `NLLB_WO_FR_MODEL` | `./src/Wo_fr_bilalfaye...`                                    | Modèle NLLB WO→FR (local ou Hub)                  |
| `NLLB_FR_WO_MODEL` | `./src/Fr-Wo_Lahad...`                                        | Modèle NLLB FR→WO (local ou Hub)                  |
| `STT_MODEL_PATH`   | `./src/stt_wolof-whisper-small-lora`                          | Modèle Whisper wolof (local)                      |
| `STT_LANGUAGE`     | `wo`                                                          | Langue forcée STT (`wo` = wolof)                  |
| `EMBED_MODEL`      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modèle embeddings                                 |
| `TTS_ENGINE`       | `oolel`                                                       | `oolel` \| `speecht5` \| `edge`                   |
| `OOLEL_TTS_REPO`   | `soynade-research/Oolel-Voices`                               | Repo HuggingFace Oolel-Voices                     |
| `WOLOF_TTS_MODEL`  | `bilalfaye/speecht5_tts-wolof-v0.2`                           | Fallback SpeechT5                                 |
| `RERANKER_MODEL`   | `cross-encoder/ms-marco-MiniLM-L-6-v2`                        | Modèle reranker                                   |
| `RERANKER_TOP_K`   | `3`                                                           | Chunks retenus après reranking                    |
| `HOST`             | `0.0.0.0`                                                     | Adresse de binding                                |
| `PORT`             | `8008`                                                        | Port du serveur                                   |
| `OCR_LANGS`        | `fra+eng`                                                     | Langues Tesseract pour l'OCR                      |

---

## Dépendances principales

```
# ML
torch, transformers==4.46.3, sentencepiece, accelerate, safetensors

# RAG
rank-bm25, sentence-transformers, chromadb, pypdf

# STT
openai-whisper, librosa>=0.10.2, av, soundfile

# API / LLM
fastapi, uvicorn, python-multipart, groq, google-generativeai

# TTS
edge-tts, torchaudio, soundfile, scipy, numpy<2.0
diffusers==0.29.0, conformer==0.3.2, s3tokenizer==0.3.0
huggingface_hub, num2words

# Divers
langdetect, pillow, datasets
```

> **Python 3.11** requis. `transformers 4.46.3` peut segfault sur Python 3.13.

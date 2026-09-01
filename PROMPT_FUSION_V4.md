# TontumaBot V4 — Prompt de création

## Le projet

Créer **TontumaBot**, un assistant conversationnel intelligent pour aider les citoyens sénégalais dans leurs démarches administratives. L'utilisateur peut poser des questions en **wolof** ou en **français**, par écrit ou par la voix, et le bot répond dans la langue de l'utilisateur avec les informations pertinentes.

---

## Ce que le bot doit faire

1. **Comprendre la question** de l'utilisateur (texte ou audio en wolof/français)
2. **Chercher l'information** dans une base de documents administratifs (procédures, tarifs, lieux, contacts)
3. **Répondre clairement** dans la langue de l'utilisateur
4. **Proposer une réponse vocale ou texte** en wolof ou français

---

## Modèles à utiliser

| Fonction | Modèle | Pourquoi |
|----------|--------|----------|
| Traduction wolof↔français | `bilalfaye/nllb-200-distilled-600M-wo-fr-en` | Spécialisé wolof, léger |
| Compréhension vocale (STT) | `openai/whisper-small` fine-tuné wolof | Transcription audio wolof → texte |
| Embeddings (indexation) | `BAAI/bge-m3` | Multilingue, haute qualité |
| Recherche de documents | ChromaDB + BM25 | Hybride : sémantique + lexical |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Réordonne finement les résultats |
| Génération de réponse | Groq `qwen/qwen3.6-27b` ou Gemini `gemini-2.0-flash` | Rapide, bon en français |
| Synthèse vocale (TTS) | `soynade-research/Oolel-Voices` (wolof) + fallback `edge-tts` (FR) | Voix naturelle wolof |

---

## Interface web FastAPI à créer

### Page principale (`/`) — Interface de chat

- Zone de chat avec bulles (utilisateur à droite, bot à gauche)
- Champ de texte pour écrire en wolof ou français
- Bouton micro pour enregistrer un message vocal (MediaRecorder API)
- Bouton photo/document pour envoyer une image (OCR)
- Toggle 🔊 pour activer/désactiver la réponse vocale
- Exemples de questions cliquables
- Affichage de la réponse avec :
  - La réponse traduite si la question était en wolof
  - Un lecteur audio si TTS activé
  - Un résumé du pipeline (langue détectée, traduction, recherche utilisée)
- Design sombre, moderne, responsive

### Page admin (`/admin`) — Gestion de la base documentaire

- Upload de fichiers (TXT, MD, PDF)
- Collage de texte brut
- Liste des documents indexés avec nombre de chunks
- Bouton supprimer par document ou tout vider
- Compteur total de chunks

### Endpoints API

| Méthode | URL | Description |
|---------|-----|-------------|
| GET | `/` | Interface chat |
| GET | `/admin` | Interface admin |
| GET | `/health` | État du serveur |
| POST | `/ask` | Question texte `{question, tts}` |
| POST | `/ask/audio` | Audio → STT → réponse |
| POST | `/ask/image` | Image → OCR → réponse |
| POST | `/admin/documents` | Ingérer un document |
| GET | `/admin/documents` | Lister les documents |
| DELETE | `/admin/documents/{id}` | Supprimer un document |

---

## Pipeline de traitement

```
Question (texte ou audio)
    ↓
Détection langue (wolof ou français)
    ↓
Si wolof → Traduction en français (NLLB)
    ↓
Détection d'intention (procédure ou orientation)
    ↓
Recherche dans la base de documents
  - Recherche hybride (BM25 + vectoriel)
  - Fusion des résultats (RRF)
  - Diversité (MMR)
  - Reranking (cross-encoder)
    ↓
Génération de la réponse (LLM)
    ↓
Si question wolof → Traduction de la réponse en wolof
    ↓
Si TTS → Synthèse vocale (Oolel-Voices ou edge-tts)
    ↓
Réponse à l'utilisateur
```

---

## Structure du projet

```
tontumabot/
├── app.py                    # API FastAPI
├── src/
│   ├── config.py            # Configuration .env
│   ├── pipeline.py          # Logique principale
│   ├── vectorstore.py       # Base vectorielle ChromaDB
│   ├── ingestion.py         # Import et découpage des documents
│   ├── language/detector.py # Détection wolof/français
│   ├── translation/nllb.py  # Traduction NLLB
│   ├── intent/router.py     # Détection d'intention
│   ├── retrieval/
│   │   ├── hybrid.py        # Recherche hybride
│   │   ├── reranker.py      # Reranking
│   │   └── mmr.py           # Diversité
│   ├── generation/llm.py    # Génération LLM
│   ├── tts/tts.py           # Synthèse vocale
│   └── input/
│       ├── stt.py           # Speech-to-Text
│       └── ocr.py           # OCR images
├── static/
│   ├── index.html           # Interface chat
│   └── admin.html           # Interface admin
├── data/
│   ├── seed_docs.py         # Documents de démo
│   └── chroma/              # Base vectorielle persistante
├── tests/                   # Tests pytest
├── requirements.txt         # Dépendances
├── .env.example             # Variables d'environnement
├── Dockerfile               # Image Docker
└── docker-compose.yml       # Orchestration Docker
```

---

## Configuration (.env)

```env
GROQ_API_KEY=votre_clé
GEMINI_API_KEY=votre_clé
LLM_PROVIDER=groq
TTS_ENGINE=oolel
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
EMBED_MODEL=BAAI/bge-m3
```

---

## Ce qu'il faut livrer

1. **Tout le code source** complet et fonctionnel
2. **Interface web** moderne et responsive (HTML/CSS/JS vanilla)
3. **Dockerfile** + **docker-compose.yml** pour le déploiement
4. **Tests** pytest pour les composants principaux
5. **README** avec instructions d'installation et d'utilisation
6. **.env.example** avec toutes les variables documentées

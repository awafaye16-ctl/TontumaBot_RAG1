# Mémoire conversationnelle et suivi de conversation — TONTOUMA-BOT

> Documentation technique et fonctionnelle du module de mémoire de session.
> Projet : TONTOUMA-BOT — assistant conversationnel hospitalier multilingue (français / wolof).
> Fichier source de référence : `src/memory/conversation.py`.

---

## 1. Titre de la fonctionnalité

**Gestion de la mémoire conversationnelle courte et suivi de conversation multi-tours (session utilisateur)**

---

## 2. Objectif de la mémoire conversationnelle

TONTOUMA-BOT répond à des questions administratives et d'orientation hospitalière à partir d'un pipeline RAG (*Retrieval-Augmented Generation*) interrogeant une base documentaire officielle. Une conversation utile ne se limite toutefois pas à des questions isolées : un usager pose souvent une première question, puis une question de suivi qui ne se comprend qu'à la lumière de la précédente (« Quels sont **ses** horaires ? »).

L'objectif de la mémoire conversationnelle est de :

- conserver, le temps d'une session, l'historique des échanges (question posée, réponse donnée) ;
- permettre au système de **reformuler** une question elliptique en question autonome avant de lancer la recherche documentaire ;
- garantir que chaque session reste **bornée dans le temps** (expiration par inactivité) et **bornée en nombre de questions** (anti-abus, maîtrise du coût LLM/traduction) ;
- rester strictement une mémoire de **court terme**, sans jamais se substituer à la base documentaire qui reste l'unique source de vérité.

---

## 3. Problème résolu

Sans mémoire conversationnelle, chaque question est traitée indépendamment. Le système ne peut pas résoudre les références implicites (pronoms, ellipses) et l'utilisateur est contraint de répéter intégralement le contexte à chaque message, ce qui est contraire à l'expérience attendue d'un assistant conversationnel.

La mémoire conversationnelle résout :

| Problème sans mémoire | Solution apportée |
|---|---|
| « Quels sont ses horaires ? » est incompréhensible seule | Reformulation à partir de l'historique → « Quels sont les horaires du service de radiologie ? » |
| Une session peut durer indéfiniment et consommer des ressources | Expiration automatique après inactivité (TTL) |
| Un utilisateur pourrait enchaîner un nombre illimité de requêtes LLM/traduction | Limite stricte de questions par session |
| Risque de confondre « ce que le bot a dit avant » avec « ce que dit le document officiel » | L'historique n'est jamais une source documentaire ; le RAG interroge toujours les documents officiels |

---

## 4. Cas d'usage utilisateur

**Cas 1 — Question de suivi (orientation)**

```
Utilisateur : Où se trouve le service de radiologie ?
Chatbot     : Le service de radiologie se trouve au premier étage.

Utilisateur : Quels sont ses horaires ?
Chatbot     : Le service de radiologie est ouvert de 8 h à 16 h.
```

**Cas 2 — Question de suivi sur un rendez-vous**

```
Utilisateur : Comment prendre rendez-vous en cardiologie ?
Chatbot     : Vous pouvez prendre rendez-vous au bureau des rendez-vous, rez-de-chaussée du bâtiment A.
Utilisateur : Quels documents dois-je apporter ce jour-là ?
Chatbot     : (comprend "ce jour-là" = le jour du rendez-vous en cardiologie évoqué précédemment)
```

**Cas 3 — Question sans lien avec l'historique**

```
Utilisateur : Quels documents pour un dossier médical ?
Chatbot     : (réponse)
Utilisateur : Où se trouvent les urgences ?
Chatbot     : (nouvelle question autonome — pas de reformulation nécessaire)
```

**Cas 4 — Question hors base documentaire**

```
Utilisateur : Quel est le taux de réussite des greffes rénales de l'hôpital ?
Chatbot     : Je ne dispose pas d'une information fiable sur ce sujet dans ma base documentaire.
```

---

## 5. Règles métier détaillées

| # | Règle |
|---|---|
| 1 | L'utilisateur déclenche la création d'une session (bouton **Démarrer** côté client). |
| 2 | Chaque session possède un identifiant unique `session_id`. |
| 3 | Une session autorise au maximum **5 questions** utilisateur (`MAX_QUESTIONS_PER_SESSION`). |
| 4 | Une question utilisateur + la réponse du chatbot forment **un tour de conversation**. |
| 5 | Le système conserve l'historique des **5 derniers tours** (`MEMORY_MAX_TURNS`). |
| 6 | Avant toute recherche RAG, le système **récupère l'historique** de la session. |
| 7 | Si la question dépend du contexte précédent, elle est **reformulée** en question autonome avant le RAG. |
| 8 | Le RAG interroge **toujours** les documents officiels, jamais uniquement l'historique. |
| 9 | L'historique conversationnel n'est **jamais** une source documentaire faisant foi. |
| 10 | En cas de contradiction entre une ancienne réponse et un document officiel retrouvé, **le document officiel prévaut**. |
| 11 | Toute activité utilisateur (nouvelle question) **réinitialise** le délai d'inactivité. |
| 12 | Sans nouvelle entrée pendant **2 minutes** (`SESSION_INACTIVITY_SECONDS` = 120), la session **expire**. |
| 13 | À l'expiration, l'historique est **supprimé automatiquement**. |
| 14 | L'utilisateur peut cliquer sur **Terminer** à tout moment pour clore la session. |
| 15 | « Terminer » supprime **immédiatement** la session et sa mémoire. |
| 16 | Après la **5ᵉ réponse**, la session se termine automatiquement et la mémoire est supprimée. |
| 17 | Toute nouvelle conversation nécessite une **nouvelle session** (nouveau `session_id`). |
| 18 | Aucune donnée conversationnelle n'est persistée sur disque dans le prototype : stockage **en RAM** uniquement, par processus. |
| 19 | Une évolution future (multi-utilisateur, multi-borne) pourra remplacer la RAM par **Redis** avec TTL. |

---

## 6. Cycle de vie d'une session

```
                     ┌───────────────────────┐
                     │        INEXISTANTE     │
                     └───────────┬───────────┘
                                 │ clic "Démarrer" /
                                 │ 1ʳᵉ question avec un session_id inédit
                                 ▼
                     ┌───────────────────────┐
              ┌─────▶│         ACTIVE         │◀─────┐
              │      │ (questions_utilisées<5) │      │
              │      └───────────┬───────────┘      │
              │                  │ question valide   │ activité →
   activité,  │                  │ (< limite)        │ TTL réinitialisé
   TTL reset  │                  ▼                    │
              │      ┌───────────────────────┐        │
              └──────┤   TOUR ENREGISTRÉ      ├────────┘
                     │ (historique += Q/R)    │
                     └───────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
   5ᵉ réponse │        clic "Terminer"      2 min sans nouvelle entrée
   envoyée    │                  │                   │
              ▼                  ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ LIMITE_ATTEINTE  │ │     TERMINÉE     │ │     EXPIRÉE      │
   │ (fermeture auto) │ │ (fermeture manu.)│ │ (TTL dépassé)    │
   └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                  │                   │
              └──────────────────┴─────────┬─────────┘
                                            ▼
                              historique supprimé de la RAM
                                            │
                                            ▼
                         une nouvelle conversation exige un
                                nouveau session_id
```

**États** :

| État | Déclencheur | Effet |
|---|---|---|
| `active` | Session créée, `questions_utilisées < 5` | Le système accepte de nouvelles questions |
| `limite_atteinte` | La 5ᵉ réponse vient d'être envoyée | Session et historique supprimés ; toute nouvelle requête sur ce `session_id` est refusée |
| `expiree` | Aucune activité pendant 120 s | Session et historique supprimés ; toute nouvelle requête sur ce `session_id` est refusée |
| `inexistante` | `session_id` jamais vu (ou déjà nettoyé) | Une nouvelle session est créée silencieusement à la première question |

---

## 7. Architecture technique

Le module de mémoire (`src/memory/conversation.py`) est un composant transverse du pipeline (`src/pipeline.py`), interrogé **avant** la recherche RAG et mis à jour **après** la génération de la réponse.

**Pipeline texte français**

```
Texte (FR) → historique session → reformulation éventuelle → recherche hybride (BM25 + vectoriel)
           → MMR → reranking (cross-encoder) → LLM → réponse FR → mémorisation du tour
```

**Pipeline texte wolof**

```
Texte (WO) → NLLB WO→FR → historique session → reformulation éventuelle (FR)
           → recherche hybride → MMR → reranking → LLM → réponse FR
           → NLLB FR→WO → réponse WO → mémorisation du tour
```

**Pipeline audio**

```
Audio → transcription (STT) → texte (FR ou WO)
      → [traduction WO→FR si nécessaire]
      → historique session → reformulation éventuelle → RAG → LLM
      → réponse FR → [traduction FR→WO si nécessaire] → synthèse vocale optionnelle
```

**Composants impliqués**

| Composant | Rôle |
|---|---|
| `memory/conversation.py` | Stocke l'historique par `session_id`, applique les règles de quota et de TTL |
| `pipeline.reformuler_question()` | Reformule une question de suivi en s'appuyant sur les 4 derniers messages de l'historique, via un appel LLM dédié |
| `pipeline.answer()` / `answer_stream()` | Orchestre : lecture de l'historique → reformulation → RAG → génération → écriture du tour |
| `app.py` (FastAPI) | Expose les routes HTTP/WebSocket et traduit les exceptions de session en codes HTTP (409, 410) |

> **Remarque sur les modèles cités dans la configuration transmise** : le LLM effectivement configuré en production sur ce projet est un modèle **Groq** (`GROQ_MODEL`, ex. `openai/gpt-oss-20b`), le STT wolof est **`M9and2M/whisper-small-wolof`** (poids locaux `wolof-whisper-small-lora/`) et le STT français est **`pierreguillou/whisper-medium-french`**. Ces valeurs diffèrent de certaines références indiquées dans le contexte de rédaction (Qwen, Wolof-HuBERT-CTC, whisper-large-v3-turbo) ; elles sont corrigées ici pour rester fidèles à la configuration réellement vérifiée du projet. Le reste de l'architecture (NLLB, embeddings, reranker, TTS Oolel-Voices) correspond à la configuration fournie.

---

## 8. Diagramme de flux (ASCII)

```
┌──────────────┐
│  Utilisateur │
└──────┬───────┘
       │ question (+ session_id)
       ▼
┌───────────────────────────────────────────────────────────────────┐
│                         API FastAPI (app.py)                       │
└──────┬───────────────────────────────────────────────────────────-┘
       ▼
┌───────────────────────────────────────────────────────────────────┐
│  1. memoire.historique_pour_question(session_id)                    │
│     ├─ session inexistante ?  → créer une nouvelle session          │
│     ├─ session close (limite) ? → erreur 409 SessionLimitReached    │
│     └─ session expirée ?      → erreur 410 SessionExpired           │
└──────┬───────────────────────────────────────────────────────────-┘
       ▼ historique (0 à 5 tours)
┌───────────────────────────────────────────────────────────────────┐
│  2. reformuler_question(question, historique)                       │
│     historique vide → question inchangée                            │
│     historique non vide → appel LLM : question autonome             │
└──────┬───────────────────────────────────────────────────────────-┘
       ▼ question autonome (français)
┌───────────────────────────────────────────────────────────────────┐
│  3. RAG : recherche hybride (BM25 + vectoriel) → MMR → reranking    │
│     Source unique de vérité = documents officiels indexés           │
└──────┬───────────────────────────────────────────────────────────-┘
       ▼ contexte documentaire (ou vide)
┌───────────────────────────────────────────────────────────────────┐
│  4. LLM (Groq) : génère la réponse strictement à partir du contexte │
│     contexte vide → réponse d'abstention explicite                  │
└──────┬───────────────────────────────────────────────────────────-┘
       ▼ réponse française [→ NLLB FR→WO si entrée wolof]
┌───────────────────────────────────────────────────────────────────┐
│  5. memoire.ajouter_tour(session_id, question, réponse)             │
│     ├─ session pleine (5/5) → ferme la session, renvoie un message  │
│     └─ sinon → historique mis à jour, dernière_activité rafraîchie  │
└──────┬───────────────────────────────────────────────────────────-┘
       ▼
┌──────────────┐
│  Utilisateur │  ← réponse + état de session (questions_restantes, statut)
└──────────────┘
```

---

## 9. Structure des données d'une session (exemple JSON)

```json
{
  "session_id": "6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d",
  "status": "active",
  "created_at": "2026-09-10T09:12:03Z",
  "last_activity": "2026-09-10T09:13:41Z",
  "questions_used": 2,
  "max_questions": 5,
  "messages": [
    {
      "role": "user",
      "content": "Où se trouve le service de radiologie ?",
      "language": "fr",
      "created_at": "2026-09-10T09:12:03Z"
    },
    {
      "role": "assistant",
      "content": "Le service de radiologie se trouve au premier étage.",
      "language": "fr",
      "created_at": "2026-09-10T09:12:05Z"
    },
    {
      "role": "user",
      "content": "Quels sont ses horaires ?",
      "language": "fr",
      "created_at": "2026-09-10T09:13:41Z"
    },
    {
      "role": "assistant",
      "content": "Le service de radiologie est ouvert de 8 h à 16 h.",
      "language": "fr",
      "created_at": "2026-09-10T09:13:43Z"
    }
  ],
  "summary": null
}
```

> **Note d'implémentation** : dans le prototype RAM (`MemoireConversation`), les champs réellement stockés par session sont `messages` (liste `{role, content}`), `questions_utilisees`, `max_questions`, `derniere_activite` et `statut`. Les champs `session_id` (utilisé comme clé du dictionnaire en mémoire), `language` par message et `created_at` par message sont représentés ici pour documenter la structure cible recommandée (utile en particulier lors d'une migration vers Redis, cf. section 23). Le champ `summary` est réservé à une évolution future (résumé automatique de session) et vaut `null` tant qu'il n'est pas implémenté.

---

## 10. Description des champs

| Champ | Type | Description |
|---|---|---|
| `session_id` | string (UUID) | Identifiant unique de la session, généré à la création. Sert de clé de recherche pour toutes les opérations sur la mémoire. |
| `messages` | array d'objets | Historique des tours de conversation, limité aux `MEMORY_MAX_TURNS` derniers tours (troncature aux plus récents). |
| `role` | string (`"user"` \| `"assistant"`) | Indique qui a émis le message : l'utilisateur ou le chatbot. |
| `content` | string | Contenu textuel du message, dans la langue de réponse (le français en interne ; le wolof est traduit en français avant stockage/traitement RAG). |
| `language` | string (`"fr"` \| `"wo"`) | Langue du message tel que perçu/produit côté utilisateur. Champ recommandé pour affichage et audit multilingue. |
| `created_at` | string (ISO 8601) | Horodatage de création du message. Champ recommandé pour le tri et le débogage. |
| `last_activity` | string (ISO 8601) | Horodatage de la dernière activité de la session ; réinitialisé à chaque tour valide, sert de référence au calcul du TTL. |
| `questions_used` | integer | Nombre de questions déjà traitées dans la session (`questions_utilisees` dans le code). |
| `max_questions` | integer | Quota maximal de questions autorisées pour la session (`MAX_QUESTIONS_PER_SESSION`, valeur de test : 5). |
| `status` | string (`"active"` \| `"limite_atteinte"` \| `"expiree"` \| `"inexistante"`) | État courant de la session (voir cycle de vie, section 6). |
| `summary` *(éventuel)* | string \| null | Résumé condensé de la session, destiné à une évolution future (ex. compression de l'historique au-delà de `MEMORY_MAX_TURNS`, ou trace d'audit courte). Non implémenté dans le prototype. |

---

## 11. Mémoire courte vs mémoire longue vs historique vs documents RAG vs métadonnées

Ces cinq notions sont fréquemment confondues ; leur séparation stricte est une règle métier du projet.

| Notion | Définition | Portée | Durée de vie | Source de vérité ? |
|---|---|---|---|---|
| **Mémoire courte (conversationnelle)** | Historique des tours d'**une** session en cours, utilisé uniquement pour désambiguïser les questions de suivi | Une session (`session_id`) | Quelques minutes (TTL 120 s), effacée à la fin de session | **Non** — jamais consultée par le RAG comme source de fait |
| **Mémoire longue** | Mémorisation persistante des préférences/faits d'un utilisateur au fil du temps, au-delà d'une session | Un utilisateur, potentiellement multi-session | Long terme (jours/mois), nécessiterait une base persistante | **Non implémentée** dans ce projet (prototype = mémoire courte uniquement) |
| **Historique conversationnel** | Synonyme opérationnel de la mémoire courte : la suite ordonnée des tours (question, réponse) d'une session | Une session | Identique à la mémoire courte | **Non** |
| **Documents RAG** | Corpus documentaire officiel (procédures administratives, fiches d'orientation, horaires) indexé dans ChromaDB | Global à l'application, partagé par toutes les sessions | Persistant (jusqu'à mise à jour/suppression par un administrateur) | **Oui — source de vérité unique** |
| **Métadonnées documentaires** | Informations structurées attachées à chaque fragment de document (titre, identifiant de document, position du fragment, date d'ingestion, etc.) | Attachées aux documents RAG | Persistant, liée au cycle de vie du document | Utilisées pour tracer/filtrer la recherche, pas des faits en elles-mêmes |

**Règle centrale** : la mémoire courte sert uniquement à *comprendre la question* (résolution de références). Elle ne sert **jamais** à *répondre* à la question — seule la recherche RAG dans les documents officiels peut fonder une réponse factuelle. En cas de divergence entre une réponse précédente du chatbot (mémoire courte) et un document officiel retrouvé, **le document prévaut toujours**.

---

## 12. Fonctionnement de la reformulation de question avant le RAG

**Objectif** : transformer une question elliptique (qui dépend du contexte précédent) en une question autonome, compréhensible sans l'historique, avant de lancer la recherche documentaire — car un moteur de recherche (BM25 + vectoriel) ne « comprend » pas les pronoms.

**Déclenchement** : la reformulation n'est tentée que si l'historique de la session n'est pas vide. Pour la première question d'une session, la question est transmise telle quelle au RAG (aucun appel de reformulation inutile).

**Méthode** : les 4 derniers messages de l'historique (2 derniers tours) sont fournis à un modèle de langage avec une consigne stricte : reformuler *uniquement* la dernière question, sans y répondre, en restituant une question complète et autonome.

**Exemple** :

| Élément | Valeur |
|---|---|
| Historique | `user: Où se trouve le service de radiologie ?` / `assistant: Le service de radiologie se trouve au premier étage.` |
| Question posée | `Quels sont ses horaires ?` |
| Question reformulée envoyée au RAG | `Quels sont les horaires du service de radiologie ?` |

**Garde-fous** :

- Si la reformulation échoue (erreur réseau, réponse vide) ou n'apporte rien, la question originale est utilisée sans blocage du pipeline — la reformulation est un **enrichissement**, jamais un point de défaillance bloquant.
- La reformulation n'agit que sur la **requête de recherche** ; elle n'altère jamais l'historique stocké (la question originale de l'utilisateur reste celle enregistrée dans `messages`).
- La question reformulée ne peut pas « inventer » un sujet absent de l'historique ; en l'absence de lien logique, le modèle est instruit de conserver la question telle quelle.

---

## 13. Exemple détaillé — conversation complète de 5 questions

Contexte : hôpital, orientation et procédures administratives.

| Tour | Question utilisateur | Question envoyée au RAG (après reformulation) | Réponse du chatbot | `questions_restantes` |
|---|---|---|---|---|
| 1 | Où se trouve le service de radiologie ? | *(inchangée — pas d'historique)* | Le service de radiologie se trouve au premier étage. | 4 |
| 2 | Quels sont ses horaires ? | Quels sont les horaires du service de radiologie ? | Le service de radiologie est ouvert de 8 h à 16 h. | 3 |
| 3 | Comment prendre rendez-vous en cardiologie ? | *(nouveau sujet, reformulation neutre)* | Vous pouvez prendre rendez-vous au bureau des rendez-vous, rez-de-chaussée du bâtiment A. | 2 |
| 4 | Quels documents dois-je apporter ? | Quels documents dois-je apporter pour un rendez-vous en cardiologie ? | Une pièce d'identité et, le cas échéant, votre carnet de santé. | 1 |
| 5 | Et pour une demande de copie de mon dossier médical ? | Quelle est la procédure pour une demande de copie de dossier médical ? | Retirez un formulaire au bureau des archives médicales, remplissez-le en précisant le motif, puis déposez-le avec une pièce d'identité. Le délai est de 5 jours ouvrables. | 0 → **session fermée automatiquement** |

À l'issue du tour 5, le chatbot ajoute un message système :

```
Vous avez atteint la limite de 5 questions. La conversation a été réinitialisée.
Démarrez une nouvelle session.
```

---

## 14. Exemple d'expiration après 2 minutes d'inactivité

```
09:00:00  Utilisateur : Où se trouve le service de radiologie ?
09:00:02  Chatbot     : Le service de radiologie se trouve au premier étage.
                        (dernière_activite = 09:00:02 ; questions_restantes = 4)

           … aucune activité pendant 2 minutes …

09:02:03  Utilisateur : Quels sont ses horaires ?
09:02:03  Chatbot     : Votre conversation a expiré après 120 secondes d'inactivité.
                        Une nouvelle session doit être démarrée.
                        [HTTP 410 Gone]
```

L'historique de la session (y compris la première question sur la radiologie) est supprimé ; l'utilisateur doit démarrer une nouvelle session pour reposer une question, même liée au même sujet.

---

## 15. Exemple de fermeture manuelle (bouton Terminer)

```
09:10:00  Utilisateur : Où se trouve le bureau des admissions ?
09:10:02  Chatbot     : Le bureau des admissions se trouve au rez-de-chaussée, bâtiment A.
                        (questions_restantes = 4)

09:10:30  Utilisateur clique sur « Terminer »
09:10:30  Système     : session et historique supprimés immédiatement (statut = "terminée")

09:11:00  Utilisateur : Et les horaires d'ouverture ?
09:11:00  Chatbot     : [HTTP 404/410] Session inexistante ou expirée. Démarrez une nouvelle session.
```

La fermeture manuelle a le même effet immédiat que l'expiration ou l'atteinte de la limite : suppression totale de l'historique en mémoire, aucune reprise possible sur ce `session_id`.

---

## 16. Exemple de fermeture après la cinquième question

Voir le tour 5 de la section 13 : dès que la réponse à la 5ᵉ question est envoyée, le système :

1. incrémente `questions_utilisees` à 5 (= `max_questions`) ;
2. marque la session comme fermée (`statut: "limite_atteinte"`) et la retire de la mémoire active ;
3. renvoie, **dans la même réponse**, un message explicite informant l'utilisateur que la limite est atteinte et qu'une nouvelle session est nécessaire ;
4. toute requête ultérieure sur ce `session_id` échoue explicitement (HTTP 409) plutôt que de silencieusement redémarrer une session vide.

---

## 17–18. API FastAPI proposée

> Les quatre endpoints ci-dessous documentent le **contrat cible** de la fonctionnalité mémoire de session, tel que demandé pour ce rapport. L'implémentation actuelle du prototype (`app.py`) réalise le même comportement fonctionnel via `POST /ask` (démarrage implicite de session à la première question), `POST /ask/audio`, `GET /session/{session_id}` et `DELETE /session/{session_id}`. La correspondance est indiquée pour chaque endpoint.

### `POST /sessions/start`

*(équivalent actuel : démarrage implicite lors du premier appel à `POST /ask` sans `session_id`, ou explicitement via `memoire.demarrer_session`)*

**Objectif** : créer une nouvelle session conversationnelle et retourner son identifiant.

**Paramètres** : aucun corps de requête requis.

**Exemple de requête**

```http
POST /sessions/start HTTP/1.1
Content-Type: application/json
```

**Exemple de réponse `200 OK`**

```json
{
  "session_id": "6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d",
  "status": "active",
  "questions_restantes": 5,
  "expiration_secondes": 120
}
```

**Erreurs possibles**

| Code | Cause |
|---|---|
| `500` | Erreur interne lors de l'initialisation de la session |

---

### `POST /sessions/{session_id}/message`

*(équivalent actuel : `POST /ask` avec `session_id` dans le corps, ou `POST /ask/audio` pour l'entrée vocale)*

**Objectif** : envoyer une question utilisateur dans une session existante et recevoir la réponse du pipeline RAG.

**Paramètres**

| Paramètre | Emplacement | Type | Requis | Description |
|---|---|---|---|---|
| `session_id` | chemin | string | oui | Identifiant de la session |
| `text` | corps | string | oui | Question de l'utilisateur |
| `language` | corps | string (`fr`\|`wo`\|`auto`) | non | Langue forcée, sinon détection automatique |
| `input_type` | corps | string (`text`\|`audio`) | non | Nature de l'entrée |

**Exemple de requête**

```http
POST /sessions/6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d/message HTTP/1.1
Content-Type: application/json

{
  "text": "Quels sont ses horaires ?",
  "language": "fr",
  "input_type": "text"
}
```

**Exemple de réponse `200 OK`**

```json
{
  "session_id": "6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d",
  "question_reformulee": "Quels sont les horaires du service de radiologie ?",
  "response": "Le service de radiologie est ouvert de 8 h à 16 h.",
  "questions_restantes": 3,
  "session_fermee": false,
  "status": "active"
}
```

**Réponse lorsque la limite vient d'être atteinte (`200 OK`, avec message explicite)**

```json
{
  "session_id": "6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d",
  "response": "Retirez un formulaire au bureau des archives médicales…",
  "questions_restantes": 0,
  "session_fermee": true,
  "message": "Vous avez atteint la limite de 5 questions. La conversation a été réinitialisée. Démarrez une nouvelle session.",
  "status": "limite_atteinte"
}
```

**Erreurs possibles**

| Code | Cause |
|---|---|
| `400` | Question vide ou langue non supportée |
| `409` | Limite de questions déjà atteinte pour cette session |
| `410` | Session expirée par inactivité |
| `500` | Erreur interne (LLM, traduction, RAG) |

---

### `GET /sessions/{session_id}/status`

*(équivalent actuel : `GET /session/{session_id}`)*

**Objectif** : consulter l'état courant d'une session sans y ajouter de tour.

**Paramètres**

| Paramètre | Emplacement | Type | Requis |
|---|---|---|---|
| `session_id` | chemin | string | oui |

**Exemple de requête**

```http
GET /sessions/6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d/status HTTP/1.1
```

**Exemple de réponse `200 OK`**

```json
{
  "session_id": "6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d",
  "status": "active",
  "questions_utilisees": 2,
  "questions_restantes": 3,
  "secondes_restantes": 87
}
```

**Réponse pour une session absente/déjà nettoyée**

```json
{
  "session_id": "…",
  "status": "inexistante",
  "questions_restantes": 0
}
```

**Erreurs possibles**

| Code | Cause |
|---|---|
| `500` | Erreur interne |

*(par choix de conception, une session introuvable renvoie un statut descriptif en `200 OK` plutôt qu'un `404`, afin qu'un client puisse afficher un état sans gérer un cas d'erreur distinct.)*

---

### `POST /sessions/{session_id}/end`

*(équivalent actuel : `DELETE /session/{session_id}`)*

**Objectif** : terminer volontairement une session (bouton **Terminer**) et supprimer immédiatement son historique.

**Paramètres**

| Paramètre | Emplacement | Type | Requis |
|---|---|---|---|
| `session_id` | chemin | string | oui |

**Exemple de requête**

```http
POST /sessions/6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d/end HTTP/1.1
```

**Exemple de réponse `200 OK`**

```json
{
  "session_id": "6f1a1e2e-2b7b-4c9a-9e21-6b8b6a2b6e7d",
  "status": "terminee",
  "message": "La session est terminée."
}
```

**Erreurs possibles**

| Code | Cause |
|---|---|
| `500` | Erreur interne lors de la suppression |

> Par idempotence, mettre fin à une session déjà expirée ou déjà terminée renvoie également `200 OK` (l'état final recherché — « aucune trace en mémoire » — est déjà atteint).

---

## 19. Proposition de classe Python `MemoireConversation`

Classe effectivement implémentée et testée dans le projet (`src/memory/conversation.py`), présentée ici comme référence :

```python
import time
from threading import RLock


class SessionLimitReached(ValueError):
    """La session a atteint son nombre maximal de questions."""


class SessionExpired(ValueError):
    """La session a expiré par inactivité et doit être redémarrée."""


class MemoireConversation:
    """Mémoire de conversation en RAM, volatile et locale au processus.

    Stocke l'historique des échanges par session_id, avec :
      - une limite de `max_questions` questions par session ;
      - un historique borné aux `max_messages_context` derniers messages ;
      - une expiration après `duree_vie_s` secondes d'inactivité.
    """

    def __init__(self, max_questions: int = 5, max_messages_context: int = 10,
                 duree_vie_s: int = 120):
        self.max_questions = max_questions
        self.max_messages_context = max_messages_context  # = MEMORY_MAX_TURNS * 2
        self.duree_vie_s = duree_vie_s
        self._sessions: dict[str, dict] = {}
        self._sessions_fermees: set[str] = set()   # limite atteinte
        self._sessions_expirees: set[str] = set()  # expirées par inactivité
        self._lock = RLock()

    def _nettoyer_sessions_expirees(self) -> None:
        now = time.time()
        expirees = [sid for sid, s in self._sessions.items()
                    if now - s["derniere_activite"] > self.duree_vie_s]
        for sid in expirees:
            del self._sessions[sid]
            self._sessions_expirees.add(sid)

    def demarrer_session(self, session_id: str) -> dict:
        with self._lock:
            self._nettoyer_sessions_expirees()
            if session_id in self._sessions_fermees:
                raise SessionLimitReached("Limite de questions atteinte pour cette session.")
            self._sessions[session_id] = {
                "messages": [], "questions_utilisees": 0,
                "max_questions": self.max_questions,
                "derniere_activite": time.time(), "statut": "active",
            }
            return {"session_id": session_id, "statut": "active",
                    "questions_restantes": self.max_questions}

    def historique_pour_question(self, session_id: str) -> list[dict]:
        """Crée la session au premier appel ; refuse une session close ou expirée."""
        with self._lock:
            self._nettoyer_sessions_expirees()
            if session_id in self._sessions_fermees:
                raise SessionLimitReached("Limite de questions atteinte pour cette session.")
            if session_id in self._sessions_expirees:
                raise SessionExpired(
                    f"Votre conversation a expiré après {self.duree_vie_s} secondes "
                    "d'inactivité. Une nouvelle session doit être démarrée."
                )
            if session_id not in self._sessions:
                self.demarrer_session(session_id)
            return list(self._sessions[session_id]["messages"])

    def ajouter_tour(self, session_id: str, question: str, reponse: str) -> dict:
        with self._lock:
            self._nettoyer_sessions_expirees()
            session = self._sessions.get(session_id)
            if session is None:
                if session_id in self._sessions_fermees:
                    raise SessionLimitReached("Limite de questions atteinte pour cette session.")
                raise SessionExpired(
                    f"Votre conversation a expiré après {self.duree_vie_s} secondes "
                    "d'inactivité. Une nouvelle session doit être démarrée."
                )
            if session["questions_utilisees"] >= self.max_questions:
                self._sessions_fermees.add(session_id)
                self._sessions.pop(session_id, None)
                raise SessionLimitReached("Limite de questions atteinte.")

            session["messages"].append({"role": "user", "content": question})
            session["messages"].append({"role": "assistant", "content": reponse})
            session["messages"] = session["messages"][-self.max_messages_context:]
            session["questions_utilisees"] += 1
            session["derniere_activite"] = time.time()

            restantes = self.max_questions - session["questions_utilisees"]
            if restantes == 0:
                self._sessions_fermees.add(session_id)
                self._sessions.pop(session_id, None)
                return {
                    "statut": "limite_atteinte", "questions_restantes": 0, "session_fermee": True,
                    "message": (f"Vous avez atteint la limite de {self.max_questions} questions. "
                                "La conversation a été réinitialisée. Démarrez une nouvelle session."),
                }
            return {"statut": "active", "questions_restantes": restantes, "session_fermee": False}

    def statut_session(self, session_id: str) -> dict:
        self._nettoyer_sessions_expirees()
        if session_id in self._sessions_fermees:
            return {"statut": "limite_atteinte", "questions_restantes": 0, "secondes_restantes": 0}
        if session_id in self._sessions_expirees:
            return {"statut": "expiree", "questions_restantes": 0, "secondes_restantes": 0}
        session = self._sessions.get(session_id)
        if session is None:
            return {"statut": "inexistante", "questions_restantes": 0}
        restantes_s = max(0, int(self.duree_vie_s - (time.time() - session["derniere_activite"])))
        return {
            "statut": session["statut"],
            "questions_utilisees": session["questions_utilisees"],
            "questions_restantes": self.max_questions - session["questions_utilisees"],
            "secondes_restantes": restantes_s,
        }

    def effacer_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._sessions_fermees.discard(session_id)
            self._sessions_expirees.discard(session_id)


# Instance unique du processus, configurée depuis .env
memoire = MemoireConversation(
    max_questions=5,          # MAX_QUESTIONS_PER_SESSION
    max_messages_context=10,  # MEMORY_MAX_TURNS(=5) * 2
    duree_vie_s=120,          # SESSION_INACTIVITY_SECONDS (alias SESSION_TTL_SECONDS)
)
```

---

## 20. Pseudo-code — traitement d'une nouvelle question

```text
FONCTION traiter_question(session_id, question_utilisateur, langue):

    # 1. Vérifier la session / le TTL
    SI session_id est marqué "fermée pour limite atteinte":
        RETOURNER erreur 409 "Limite de questions atteinte pour cette session."

    SI session_id est marqué "expirée":
        RETOURNER erreur 410 "Session expirée par inactivité. Démarrez une nouvelle session."

    SI session_id est inconnu:
        créer une nouvelle session (questions_utilisees = 0, historique = [])

    # 2. Récupérer l'historique
    historique = derniers MEMORY_MAX_TURNS tours de la session

    # 3. Reformuler la question si nécessaire
    SI historique n'est pas vide:
        question_recherche = reformuler_via_LLM(question_utilisateur, historique)
    SINON:
        question_recherche = question_utilisateur

    # 3bis. Traduction si entrée en wolof
    SI langue == "wo":
        question_recherche = traduire_wolof_vers_francais(question_recherche)

    # 4. Lancer le RAG (source de vérité = documents officiels, jamais l'historique)
    contexte = recherche_hybride(question_recherche)      # BM25 + vectoriel
    contexte = MMR(contexte)                               # diversité
    contexte = reranking(contexte, question_recherche)      # pertinence finale

    # 5. Générer la réponse
    SI contexte est vide:
        reponse = "Je ne dispose pas d'une information fiable sur ce sujet."
    SINON:
        reponse = generer_LLM(question_recherche, contexte)  # strictement fondée sur le contexte

    SI langue == "wo":
        reponse_finale = traduire_francais_vers_wolof(reponse)
    SINON:
        reponse_finale = reponse

    # 6. Ajouter le tour à l'historique + mettre à jour la dernière activité
    ESSAYER:
        etat = memoire.ajouter_tour(session_id, question_utilisateur, reponse_finale)
    ATTRAPER SessionLimitReached / SessionExpired:
        # cas limite : la session a expiré/s'est fermée pendant le traitement
        RETOURNER erreur correspondante (409 / 410)

    # 7. Fermer la session si la limite est atteinte (déjà géré par ajouter_tour)
    SI etat.session_fermee == vrai:
        inclure etat.message dans la réponse au client

    RETOURNER {
        reponse: reponse_finale,
        questions_restantes: etat.questions_restantes,
        session_fermee: etat.session_fermee,
        message: etat.message (si présent)
    }
```

---

## 21. Sécurité et confidentialité

- **Aucune persistance disque** dans le prototype : l'historique n'existe qu'en RAM, dans le processus du serveur applicatif.
- **Suppression effective en fin de session** : expiration, clic sur « Terminer » ou atteinte de la 5ᵉ question suppriment intégralement l'historique associé au `session_id` — aucune trace résiduelle.
- **Minimisation des données** : seules les données strictement nécessaires au suivi de conversation (texte des tours, compteur, horodatage d'activité) sont conservées ; aucune donnée d'identification personnelle n'est requise pour utiliser le chatbot (pas de nom, pas de numéro de dossier patient dans la mémoire conversationnelle elle-même).
- **Journalisation prudente** : les journaux applicatifs ne doivent **pas** écrire le contenu intégral des échanges utilisateur (questions/réponses) en clair ; se limiter à des métriques (type d'erreur, latence, identifiants tronqués) évite qu'une information potentiellement sensible ne persiste dans des fichiers de logs.
- **Isolation par session** : un `session_id` ne donne accès qu'à sa propre mémoire ; aucune fonction n'expose l'historique d'une session à partir d'un autre identifiant.
- **Recommandations de déploiement** : chiffrement du transport (HTTPS/TLS obligatoire en production), contrôle d'accès sur les routes d'administration (`/admin/*`, déjà protégées par clé API ou restriction réseau dans ce projet), limitation de débit (*rate limiting*) pour prévenir les abus sur la création de sessions.

---

## 22. Limites de l'approche RAM

| Limite | Explication |
|---|---|
| **Perte des sessions au redémarrage** | La mémoire vit dans le processus Python ; tout redémarrage (déploiement, crash, `--reload`) efface toutes les sessions actives sans préavis pour l'utilisateur. |
| **Non-partage entre plusieurs processus** | Si l'application est répliquée (plusieurs workers Uvicorn/Gunicorn, plusieurs conteneurs), chaque processus a sa **propre** mémoire : une requête peut arriver sur un processus qui ne connaît pas la session créée par un autre. |
| **Difficulté pour un déploiement multi-borne** | Dans un contexte hospitalier avec plusieurs bornes physiques (accueil, différents étages), une session commencée sur une borne ne peut pas être poursuivie sur une autre sans mémoire partagée. |
| **Risque de saturation mémoire** | En l'absence de purge stricte, un grand nombre de sessions actives simultanées (ou un TTL mal configuré) peut faire croître indéfiniment la mémoire du processus ; le nettoyage n'a lieu qu'au moment d'un accès à une session (pas de tâche de fond dédiée dans le prototype). |

---

## 23. Évolution future avec Redis

Pour un déploiement multi-utilisateur ou multi-borne, la mémoire RAM peut être remplacée par **Redis**, sans changer la logique métier (mêmes règles, même signature de fonctions) :

- Chaque session est stockée sous la clé **`session:{session_id}`**, valeur sérialisée en JSON (mêmes champs que la section 9).
- **TTL de 120 secondes** posé sur la clé à la création (`SET … EX 120` ou `SETEX`).
- **Renouvellement du TTL après chaque question valide** (`EXPIRE session:{session_id} 120`), reproduisant la réinitialisation du délai d'inactivité.
- **Suppression automatique** par Redis lui-même à l'expiration de la clé — plus besoin d'une purge applicative périodique.
- Redis étant un magasin partagé accessible par tous les processus/bornes, la session devient **cohérente quel que soit le processus ou la borne physique** qui traite la requête suivante : meilleure compatibilité multi-borne et haute disponibilité.
- La fermeture manuelle (« Terminer ») ou l'atteinte de la limite de questions se traduisent simplement par un `DEL session:{session_id}` immédiat, comme en RAM.

```python
# Exemple d'esquisse (asynchrone), à titre d'illustration
await redis.set(f"session:{session_id}", json.dumps(session), ex=120)
...
await redis.expire(f"session:{session_id}", 120)   # à chaque tour valide
...
await redis.delete(f"session:{session_id}")         # Terminer / limite atteinte
```

> Le sous-projet `Borne_V1` de ce dépôt contient déjà une ébauche de service de session basée sur `redis.asyncio`, avec la même politique de TTL glissant, prête à être branchée sur le véritable pipeline RAG.

---

## 24. Stratégie de test

| Niveau | Ce qui est testé | Exemple |
|---|---|---|
| **Unitaire — `MemoireConversation`** | Création de session, comptage des questions, fermeture à la limite, expiration explicite, nettoyage des sessions closes/expirées | Une session à `max_questions=2` se ferme après 2 tours ; `ajouter_tour` sur une session expirée lève `SessionExpired`, pas une erreur générique |
| **Unitaire — reformulation** | La reformulation n'est appelée que si l'historique n'est pas vide ; un échec de reformulation ne bloque pas le pipeline | Historique vide → question inchangée, aucun appel LLM |
| **Intégration — pipeline** | L'état de session (`session`) est bien propagé dans la réponse de `pipeline.answer()` / `answer_stream()` ; la limite lève bien l'exception attendue au tour suivant | Séquence de 2 appels avec le même `session_id` sur une mémoire à `max_questions=2` : le 1ᵉʳ retourne `session_fermee: false`, le 2ᵉ `session_fermee: true` avec message, le 3ᵉ lève `SessionLimitReached` |
| **API (bout en bout)** | Codes HTTP renvoyés (`409` limite, `410` expiration), propagation du `session_id`, cohérence WebSocket/REST | `POST /ask` deux fois avec le même `session_id` sur une base réelle → 200/200/409 |
| **Cas limite (race condition)** | Une requête lente (chargement de modèle) qui dépasse le TTL pendant son traitement doit renvoyer une expiration explicite, pas une erreur générique | Simulation d'expiration entre la lecture de l'historique et l'écriture du tour → `SessionExpired`, pas `ValueError` générique |
| **Non-régression fonctionnelle** | Suivi de conversation réel : la question de suivi est bien reformulée et la réponse reste fondée sur les documents officiels | Scénario radiologie (section 13), vérifié avec les vrais modèles (LLM, RAG) en environnement de test |

---

## 25. Critères d'acceptation

- [ ] Une session est créée avec un `session_id` unique dès la première interaction.
- [ ] Une session accepte au maximum **5 questions** ; la 6ᵉ tentative est explicitement refusée (409).
- [ ] L'historique conservé ne dépasse jamais **5 tours** (troncature aux plus récents).
- [ ] Une question de suivi ambiguë (ex. « ses horaires ») est correctement reformulée en question autonome avant la recherche RAG.
- [ ] Le RAG interroge toujours les documents officiels ; en l'absence de document pertinent, la réponse est une abstention explicite, jamais une réponse inventée.
- [ ] En cas de contradiction entre l'historique et un document officiel, la réponse suit le document officiel.
- [ ] Toute question valide réinitialise le délai d'inactivité de la session.
- [ ] Une session inactive pendant **120 secondes** expire automatiquement et son historique est supprimé ; toute requête ultérieure sur ce `session_id` est explicitement refusée (410), sans redémarrage silencieux.
- [ ] Le clic sur « Terminer » supprime immédiatement la session et son historique.
- [ ] Après la 5ᵉ réponse, la session se ferme automatiquement et l'utilisateur en est informé dans la réponse elle-même.
- [ ] Aucune donnée conversationnelle n'est écrite sur disque par le module de mémoire.
- [ ] Les journaux applicatifs ne contiennent pas le texte intégral des échanges utilisateur.

---

## 26. Formulation académique (prête à insérer dans un mémoire/rapport)

> Afin d'assurer un suivi de conversation cohérent tout en maîtrisant les ressources consommées, TONTOUMA-BOT intègre une mémoire conversationnelle de court terme, propre à chaque session utilisateur. Cette mémoire conserve, en mémoire vive et pour la durée de la session uniquement, l'historique des cinq derniers tours d'échange (question et réponse). Elle permet au système de résoudre les références implicites d'une question de suivi — par exemple un pronom renvoyant au sujet évoqué précédemment — en la reformulant, à l'aide du modèle de langage, en une question autonome avant de lancer la recherche documentaire. Cette mémoire n'est cependant jamais considérée comme une source d'information factuelle : seule la base documentaire officielle, interrogée par le pipeline de recherche augmentée (RAG), fait foi ; en cas de divergence entre un échange antérieur et un document officiel nouvellement retrouvé, ce dernier prévaut systématiquement. Chaque session est bornée à cinq questions et expire automatiquement après deux minutes d'inactivité, deux mécanismes garantissant à la fois une expérience conversationnelle maîtrisée et une consommation raisonnable des ressources de calcul (appels au modèle de langage, traduction, synthèse vocale). Dans sa version prototype, cette mémoire est stockée exclusivement en mémoire vive, sans persistance disque, et supprimée immédiatement à la clôture de la session — que celle-ci résulte d'une action explicite de l'utilisateur, de l'atteinte du quota de questions ou d'une inactivité prolongée. Une évolution vers un magasin de données partagé tel que Redis, avec une politique de durée de vie (TTL) glissante équivalente, est envisagée pour permettre un déploiement multi-utilisateur et multi-borne, sans remise en cause de la logique métier établie durant cette phase de prototypage.

---

*Document généré pour le projet TONTOUMA-BOT. Fichier source de la fonctionnalité documentée : `src/memory/conversation.py`. Dernière vérification technique des règles métier (5 questions / 120 s / 5 tours) réalisée par exécution réelle du pipeline et de la suite de tests de non-régression du projet.*

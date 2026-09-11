"""Vectorstore V3 — ChromaDB + embeddings + BM25 hybride + MMR.

Pipeline de recherche :
  1. Hybrid search   BM25 (lexical) + vectoriel (sémantique)  → top-K candidats
  2. MMR             Maximum Marginal Relevance               → diversité, anti-redondance
  3. Reranker        cross-encoder                            → tri final par pertinence

Toutes les fonctions de recherche retournent des tuples
  (chunk_id, texte, score, metadata)
pour que le pipeline puisse tracer chaque étape.
"""
import hashlib
import json
import os
import sys
from threading import RLock
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import chromadb
import numpy as np

EMBED_MODEL = settings.EMBED_MODEL
COLLECTION  = "tontuma_v3"
DB_DIR      = os.path.join(settings.BASE_DIR, "data", "chroma")

_embedder   = None
_client     = None
_collection = None
_init_lock = RLock()
_write_lock = RLock()


# =============================================================================
#  Embedding
# =============================================================================

def get_embedder() -> "SentenceTransformer":
    global _embedder
    with _init_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer  # import lazy
            print(f"[VS] Chargement embedder ({EMBED_MODEL})...")
            _embedder = SentenceTransformer(EMBED_MODEL)
        return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def _embed_single(text: str) -> list[float]:
    return _embed([text])[0]


# =============================================================================
#  Collection ChromaDB
# =============================================================================

def get_client() -> chromadb.PersistentClient:
    global _client
    with _init_lock:
        if _client is None:
            os.makedirs(DB_DIR, exist_ok=True)
            _client = chromadb.PersistentClient(path=DB_DIR)
        return _client


def get_collection():
    """Open safely; the historical purge policy below is no longer applied.

    Only Chroma's explicit missing-collection errors permit creation. A model
    mismatch (including missing metadata) requires an operator-managed migration
    to a separate collection; existing data is never deleted automatically.
    """
    global _collection
    with _init_lock:
        if _collection is not None:
            return _collection
        client = get_client()
        hnsw_meta = {"hnsw:space": "cosine"}
        missing_errors = tuple(
            error for name in ("NotFoundError", "InvalidCollectionException")
            if isinstance(error := getattr(chromadb.errors, name, None), type)
            and issubclass(error, Exception)
        )
        try:
            collection = client.get_collection(COLLECTION)
        except missing_errors:
            collection = client.create_collection(
                COLLECTION,
                metadata={**hnsw_meta, "embed_model": EMBED_MODEL},
            )
        # Si le modèle d'embedding a changé → on purge pour rester cohérent
        stored_model = (collection.metadata or {}).get("embed_model")
        if stored_model != EMBED_MODEL:
            raise RuntimeError(
                f"Collection {COLLECTION!r} in {DB_DIR!r} uses embed_model "
                f"{stored_model!r}, but the configured model is {EMBED_MODEL!r}. "
                "No data was changed. Restore the original EMBED_MODEL or back "
                "up the database and explicitly re-ingest into a separate "
                "collection/database. Missing metadata must be verified before "
                "an operator-managed migration."
            )
        _collection = collection
        return _collection


# =============================================================================
#  Écriture
# =============================================================================

def add_documents(chunks: list[str], metadatas: list[dict]) -> int:
    """Ajoute (ou met à jour) des chunks dans ChromaDB.

    L'ID est un hash SHA-256 du texte et des métadonnées stables, hors dates
    d'ingestion. Le retour compte les chunks uniques soumis, pas les nouveautés.
    Les anciens IDs issus d'une autre stratégie ne sont pas migrés ni supprimés.
    """
    if len(chunks) != len(metadatas):
        raise ValueError("chunks and metadatas must have the same length")
    if not chunks:
        return 0
    rows = {}
    for chunk, metadata in zip(chunks, metadatas):
        if not isinstance(chunk, str) or not chunk.strip():
            raise ValueError("Each chunk must be a non-empty string")
        if not isinstance(metadata, dict):
            raise ValueError("Each chunk metadata must be a dictionary")
        stable_metadata = {
            key: value for key, value in metadata.items()
            if key not in {"added", "timestamp", "ingested_at", "updated_at"}
        }
        payload = json.dumps(
            [chunk, stable_metadata], sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        )
        cid = "doc-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        rows[cid] = (chunk, dict(metadata))
    documents = [row[0] for row in rows.values()]
    coll = get_collection()
    embeddings = _embed(documents)
    with _write_lock:
        coll.upsert(
            ids=list(rows), documents=documents, embeddings=embeddings,
            metadatas=[row[1] for row in rows.values()],
        )
    return len(rows)


# =============================================================================
#  Recherche vectorielle pure
# =============================================================================

def vector_search(
    query: str,
    k: int = 10,
    filter_meta: Optional[dict] = None,
) -> list[tuple[str, str, float, dict]]:
    """Recherche vectorielle cosinus dans ChromaDB.

    Returns:
        Liste de (chunk_id, texte, score_similarité, metadata), triée par score desc.
    """
    if k <= 0 or not query.strip():
        return []
    coll = get_collection()
    total = coll.count()
    if total == 0:
        return []

    kwargs = {"query_embeddings": [_embed_single(query)], "n_results": min(k, total),
              "include": ["documents", "metadatas", "distances"]}
    if filter_meta:
        kwargs["where"] = filter_meta

    res = coll.query(**kwargs)
    out = []
    for doc, meta, dist, cid in zip(
        res["documents"][0], res["metadatas"][0],
        res["distances"][0],  res["ids"][0],
    ):
        out.append((cid, doc, 1.0 - dist, meta or {}))  # distance cosinus → similarité
    return out


# =============================================================================
#  Recherche hybride BM25 + vectorielle
# =============================================================================

def hybrid_search(
    query: str,
    k: int   = 10,
    alpha: float = 0.4,
) -> list[tuple[str, str, float, dict]]:
    """Fusionne BM25 (lexical) et vectoriel (sémantique).

    score_final = alpha / (60 + rang_BM25) + (1 - alpha) / (60 + rang_vectoriel)
    Un résultat absent d'une liste ne reçoit aucune contribution de cette liste.
    Ces scores RRF ne sont pas des probabilités ni des scores de confiance.

    alpha = 0.4 → légère préférence au sémantique (meilleur sur les questions
    administratives paraphrasées).

    Returns:
        Liste de (chunk_id, texte, score_fusionné, metadata), top-K.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if k <= 0 or not query.strip():
        return []
    coll = get_collection()
    if coll.count() == 0:
        return []

    from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

    with _write_lock:
        all_data = coll.get(include=["documents", "metadatas"])
    ids       = all_data["ids"]
    texts     = all_data["documents"]
    metas     = all_data["metadatas"]

    # BM25 sur tous les chunks
    retriever  = HybridRetriever(texts)
    bm25_ranked = retriever._bm25_search(query, k=k * 3)

    # Vectoriel
    id_to_pos = {cid: i for i, cid in enumerate(ids)}
    vec_ranked = [
        (id_to_pos[cid], score)
        for cid, _, score, _ in vector_search(query, k=k * 3)
        if cid in id_to_pos
    ]

    # Fusion
    ranked = reciprocal_rank_fusion(bm25_ranked, vec_ranked, alpha=alpha)
    return [(ids[idx], texts[idx], score, metas[idx] or {}) for idx, score in ranked[:k]]


# =============================================================================
#  MMR — Maximum Marginal Relevance
# =============================================================================

def mmr_search(
    query: str,
    k: int      = 8,
    fetch_k: int = 20,
    lambda_mmr: float = 0.6,
    candidates: Optional[list[tuple[str, str, float, dict]]] = None,
) -> list[tuple[str, str, float, dict]]:
    """Récupère fetch_k candidats par hybride puis sélectionne k chunks
    par MMR pour maximiser pertinence et diversité.

    MMR(d) = lambda * sim(query, d) - (1 - lambda) * max_sim(d, already_selected)

    lambda_mmr = 0.6 : légèrement biaisé vers la pertinence,
                       mais avec bonne diversité (évite les chunks redondants).

    Returns:
        Liste de (chunk_id, texte, similarité_cosinus, metadata), k éléments distincts.
        Les candidats fournis sont réutilisés sans nouvelle recherche, même vides.
    """
    if not 0 <= lambda_mmr <= 1:
        raise ValueError("lambda_mmr must be between 0 and 1")
    if k <= 0 or fetch_k <= 0 or not query.strip():
        return []
    if candidates is None:
        candidates = hybrid_search(query, k=fetch_k)
    else:
        candidates = list(candidates[:fetch_k])
    if not candidates:
        return []

    # ── Déduplication par texte exact avant MMR ───────────────────────────
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()
    deduped: list = []
    for item in candidates:
        txt = item[1].strip()
        if txt and txt not in seen_texts and item[0] not in seen_ids:
            seen_texts.add(txt)
            seen_ids.add(item[0])
            deduped.append(item)
    candidates = deduped
    if not candidates:
        return []

    embedder    = get_embedder()
    q_vec       = np.array(embedder.encode([query], normalize_embeddings=True)[0])
    cand_texts  = [c[1] for c in candidates]
    cand_vecs   = embedder.encode(cand_texts, normalize_embeddings=True)

    selected_idx  : list[int]   = []
    selected_vecs : list[np.ndarray] = []
    remaining_idx : list[int]   = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        best_idx   = -1
        best_score = -float("inf")

        for i in remaining_idx:
            relevance = float(np.dot(q_vec, cand_vecs[i]))
            if selected_vecs:
                redundancy = max(
                    float(np.dot(cand_vecs[i], sv)) for sv in selected_vecs
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        selected_idx.append(best_idx)
        selected_vecs.append(cand_vecs[best_idx])
        remaining_idx.remove(best_idx)

    return [
        (candidates[i][0], candidates[i][1], float(np.dot(q_vec, cand_vecs[i])), candidates[i][3])
        for i in selected_idx
    ]


# =============================================================================
#  Recherche filtrée par métadonnée (pour l'intention "orientation")
# =============================================================================

def get_document_chunks(document_id: str) -> dict[int, tuple[str, str, dict]]:
    """Retourne {chunk_index: (chunk_id, texte, metadata)} pour un document.

    Utilisé pour étendre le contexte RAG aux chunks voisins d'un passage retenu,
    afin de ne pas couper une procédure/liste au milieu d'une frontière de chunk.
    """
    if not document_id:
        return {}
    coll = get_collection()
    data = coll.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    result = {}
    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        idx = (meta or {}).get("chunk_index")
        if isinstance(idx, int):
            result[idx] = (cid, doc, meta or {})
    return result


def filtered_search(
    query: str,
    filter_meta: dict,
    k: int = 5,
) -> list[tuple[str, str, float, dict]]:
    """Restreint la recherche aux chunks ayant des métadonnées spécifiques.

    Exemple : filtered_search(query, {"document_id": "abc123"}, k=5)
    """
    return vector_search(query, k=k, filter_meta=filter_meta)


# =============================================================================
#  Gestion des documents
# =============================================================================

def all_documents() -> list[dict]:
    """Retourne la liste des documents indexés (dédoublonnés par document_id)."""
    coll = get_collection()
    data = coll.get(include=["documents", "metadatas"])
    docs: dict = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        did = meta.get("document_id", "?")
        if did not in docs:
            docs[did] = {
                "id":     did,
                "title":  meta.get("title", did),
                "source": meta.get("source", ""),
                "chunks": 0,
                "added":  meta.get("added", ""),
            }
        docs[did]["chunks"] += 1
    return list(docs.values())


def delete_document(document_id: str) -> int:
    with _write_lock:
        coll = get_collection()
        data = coll.get(where={"document_id": document_id})
        ids = data["ids"]
        if ids:
            coll.delete(ids=ids)
        return len(ids)


def clear_all() -> int:
    with _write_lock:
        coll = get_collection()
        n = coll.count()
        if n:
            coll.delete(ids=coll.get()["ids"])
        return n


def count() -> int:
    return get_collection().count()

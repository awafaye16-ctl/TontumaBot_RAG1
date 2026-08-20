"""ChromaDB vector store V2 — embeddings + BM25 hybride + reranker."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = settings.EMBED_MODEL
COLLECTION = "tontuma_v2"
DB_DIR = os.path.join(settings.BASE_DIR, "data", "chroma")

_embedder = None
_client = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Chargement embedding ({EMBED_MODEL})...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def get_client():
    global _client
    if _client is None:
        os.makedirs(DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = get_client()
    meta = {"hnsw:space": "cosine"}
    try:
        _collection = client.get_collection(COLLECTION)
        existing_meta = _collection.metadata or {}
        if existing_meta.get("embed_model") != EMBED_MODEL:
            client.delete_collection(COLLECTION)
            _collection = client.create_collection(
                COLLECTION, metadata={**meta, "embed_model": EMBED_MODEL}
            )
    except Exception:
        _collection = client.create_collection(
            COLLECTION, metadata={**meta, "embed_model": EMBED_MODEL}
        )
    return _collection


def add_documents(chunks: list[str], metadatas: list[dict]) -> int:
    coll = get_collection()
    ids = [
        "doc-" + hashlib.md5((c + "|" + str(m)).encode()).hexdigest()[:16]
        for c, m in zip(chunks, metadatas)
    ]
    embeddings = _embed(chunks)
    coll.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def vector_search(query: str, k: int = 5) -> list[tuple[str, str, float, dict]]:
    coll = get_collection()
    if coll.count() == 0:
        return []
    q_vec = _embed([query])[0]
    res = coll.query(query_embeddings=[q_vec], n_results=k,
                     include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist, cid in zip(
        res["documents"][0], res["metadatas"][0],
        res["distances"][0], res["ids"][0]
    ):
        out.append((cid, doc, 1.0 - dist, meta))
    return out


def hybrid_search(query: str, k: int = 5, alpha: float = 0.4) -> list[tuple[str, str, float, dict]]:
    coll = get_collection()
    if coll.count() == 0:
        return []
    from retrieval.hybrid import HybridRetriever
    all_docs = coll.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    texts = all_docs["documents"]
    metas = all_docs["metadatas"]

    hybrid = HybridRetriever(texts)
    bm25_hits = dict(hybrid._bm25_search(query, k=k * 2))
    id_to_pos = {cid: i for i, cid in enumerate(ids)}
    vec_hits = {}
    for cid, _, s, _ in vector_search(query, k=k * 2):
        vec_hits[id_to_pos[cid]] = s

    results = []
    for idx in set(bm25_hits) | set(vec_hits):
        s_b = bm25_hits.get(idx, 0.0)
        s_v = vec_hits.get(idx, 0.0)
        results.append((ids[idx], texts[idx], alpha * s_b + (1 - alpha) * s_v, metas[idx]))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:k]


def all_documents() -> list[dict]:
    coll = get_collection()
    data = coll.get(include=["documents", "metadatas"])
    docs: dict = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        did = meta.get("document_id", "?")
        if did not in docs:
            docs[did] = {"id": did, "title": meta.get("title", did),
                         "chunks": 0, "added": meta.get("added", "")}
        docs[did]["chunks"] += 1
    return list(docs.values())


def delete_document(document_id: str) -> int:
    coll = get_collection()
    data = coll.get(where={"document_id": document_id})
    ids = data["ids"]
    if ids:
        coll.delete(ids=ids)
    return len(ids)


def clear_all() -> int:
    coll = get_collection()
    n = coll.count()
    if n:
        coll.delete(ids=coll.get()["ids"])
    return n


def count() -> int:
    return get_collection().count()

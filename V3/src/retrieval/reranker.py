"""Reranker : cross-encoder pour reranking des résultats du retriever."""
import os
import sys
from threading import RLock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

_model = None
_model_lock = RLock()


def load_model():
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import CrossEncoder  # import lazy
        print(f"[Reranker] Chargement ({settings.RERANKER_MODEL})...")
        _model = CrossEncoder(settings.RERANKER_MODEL)
        print("[Reranker] Chargé.")
        return _model


def rerank(query: str, documents: list[str], top_k: int = None) -> list[tuple[int, float, str]]:
    """Rerank `documents` par rapport à `query`.
    Retourne [(index_original, score, texte)] triés par pertinence décroissante.
    Les scores bruts, éventuellement négatifs, ne sont pas des probabilités et
    ne sont pas filtrés par un seuil arbitraire d'abstention.
    """
    if not documents:
        return []
    if top_k is None:
        top_k = settings.RERANKER_TOP_K
    if top_k <= 0:
        return []
    model = load_model()
    pairs = [(query, doc) for doc in documents]
    with _model_lock:
        scores = model.predict(pairs)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(idx, float(score), documents[idx]) for idx, score in indexed[:top_k]]

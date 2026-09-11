"""Recherche hybride BM25 + vectoriel par fusion réciproque des rangs (RRF)."""
import re
import unicodedata
from threading import RLock

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", text)


def reciprocal_rank_fusion(
    bm25_ranked: list[tuple[int, float]],
    vector_ranked: list[tuple[int, float]],
    alpha: float = 0.5,
    rank_constant: int = 60,
) -> list[tuple[int, float]]:
    """Weighted RRF; score magnitudes never mix across retrieval methods.

    Input lists must be best-first. Missing ranks contribute zero; returned
    scores are ordering signals, not calibrated probabilities of relevance.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores = {}
    for ranked, weight in ((bm25_ranked, alpha), (vector_ranked, 1 - alpha)):
        if weight == 0:
            continue
        seen = set()
        for rank, (idx, _) in enumerate(ranked, 1):
            if idx not in seen:
                scores[idx] = scores.get(idx, 0.0) + weight / (rank_constant + rank)
                seen.add(idx)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(self, documents: list[str], embedder=None, weights=(0.5, 0.5)):
        self.documents = list(documents)
        self.embedder = embedder
        self._tokenized = [tokenize(d) for d in self.documents]
        self.bm25 = BM25Okapi(self._tokenized) if any(self._tokenized) else None
        self.weights = weights
        self._embeddings = None
        self._embedding_lock = RLock()

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if k <= 0 or self.bm25 is None:
            return []
        q_tokens = set(tokenize(query))
        scores = self.bm25.get_scores(list(q_tokens))
        matches = [
            (idx, float(s))
            for idx, s in enumerate(scores)
            if q_tokens & set(self._tokenized[idx])
        ]
        matches.sort(key=lambda x: (-x[1], x[0]))
        return matches[:k]

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.embedder is None or not self.documents or k <= 0:
            return []
        import numpy as np
        with self._embedding_lock:
            if self._embeddings is None:
                self._embeddings = np.asarray(self.embedder(self.documents), dtype=float)
            q_vec = np.asarray(self.embedder([query])[0], dtype=float)
            norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_vec)
            scores = np.divide(
                np.dot(self._embeddings, q_vec), norms,
                out=np.zeros(len(self.documents)), where=norms != 0,
            )
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: (-x[1], x[0]))
        return [(idx, float(s)) for idx, s in ranked[:k]]

    def search(self, query: str, k: int = 5, alpha: float = None) -> list[tuple[int, float]]:
        if alpha is None:
            alpha = self.weights[0]
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if k <= 0 or not query.strip():
            return []
        bm25_ranked = self._bm25_search(query, k=k * 3)
        vec_ranked = self._vector_search(query, k=k * 3)
        return reciprocal_rank_fusion(bm25_ranked, vec_ranked, alpha=alpha)[:k]

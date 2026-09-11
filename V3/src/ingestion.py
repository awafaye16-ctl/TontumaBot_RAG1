"""Ingestion V3 — Pipeline RAG rigoureux.

Étapes :
  1. Extraction texte  (PDF, TXT, MD)
  2. Chunking sémantique  (sur frontières de phrases, taille + overlap configurables)
  3. Métadonnées enrichies par chunk  (document_id, titre, source, index, nb_tokens, date)
  4. Embedding + stockage ChromaDB  (via vectorstore.add_documents)

Le chunking respecte les frontières de phrases (.!?) et les titres Markdown (##)
pour éviter de couper un contexte sémantique au milieu d'une idée.
"""
import hashlib
import json
import os
import re
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorstore  # noqa: E402

# ── Paramètres de chunking ────────────────────────────────────────────────
# CHUNK_SIZE calé sur la taille typique d'une section/procédure complète
# (ex. une fiche de procédure administrative fait ~800-1200 caractères) pour
# limiter le nombre de chunks par section. Le pipeline RAG étend malgré tout
# chaque passage retenu à ses chunks voisins (voir pipeline._expand_context)
# pour ne jamais couper une liste d'étapes au milieu.
CHUNK_SIZE    = 1000  # caractères max par chunk
CHUNK_OVERLAP = 150   # chevauchement pour conserver le contexte entre chunks
MIN_CHUNK_LEN = 40    # on ignore les micro-fragments vides


# =============================================================================
#  1. Extraction de texte
# =============================================================================

def extract_text(path: str) -> str:
    """Extrait le texte brut d'un fichier (TXT, MD, PDF)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf requis pour les PDF : pip install pypdf")
    reader = PdfReader(path)
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


# =============================================================================
#  2. Chunking sémantique
# =============================================================================

def _normalize(text: str) -> str:
    """Normalise les espaces et les lignes vides multiples."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    """Découpe en phrases sur .!? et titres Markdown."""
    # Titres Markdown → frontière de phrase
    text = re.sub(r"(?m)^(?=#{1,6}\s+)", "\n\n", text)
    # Séparation sur .!? suivis d'un espace ou newline
    parts = re.split(r"(?<=[.!?…])\s+|\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> list[str]:
    """Découpe le texte en chunks cohérents.

    La borne de caractères est stricte, même pour une phrase ou un mot long.
    Les frontières de phrases, paragraphes, titres puis mots sont privilégiées.
    Le chevauchement est au plus `overlap` caractères et peut être réduit pour
    commencer sur un mot entier. Aucun fragment non vide n'est abandonné.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be an integer satisfying 0 <= overlap < chunk_size")
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentence_ends = [m.end() for m in re.finditer(
        r"(?<=[.!?…])\s+|\n\n+|\n(?=#{1,6}\s)", text,
    )]
    word_ends = [m.end() for m in re.finditer(r"\s+", text)]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            minimum_end = start + max(overlap + 1, chunk_size // 2)
            for boundaries in (sentence_ends, word_ends):
                pos = bisect_right(boundaries, end) - 1
                if pos >= 0 and boundaries[pos] >= minimum_end:
                    end = boundaries[pos]
                    break
        current = text[start:end].strip()
        if current:
            # Sauvegarde le chunk courant
            chunks.append(current)
        if end == len(text):
            break
        # Repart avec le chevauchement
        next_start = max(start + 1, end - overlap)
        if overlap and next_start > 0 and not text[next_start - 1].isspace():
            boundary = bisect_right(word_ends, next_start - 1)
            if boundary < len(word_ends) and word_ends[boundary] <= end:
                next_start = word_ends[boundary]
        start = next_start

    # Filtre les micro-fragments
    return [c for c in chunks if c]


# =============================================================================
#  3. Métadonnées enrichies
# =============================================================================

def _make_metadata(
    document_id: str,
    title: str,
    source: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    added: str,
) -> dict:
    """Construit les métadonnées d'un chunk.

    Champs :
      document_id   — identifiant unique du document parent
      title         — titre du document
      source        — nom du fichier ou 'manual'
      chunk_index   — position du chunk dans le document (0-based)
      total_chunks  — nombre total de chunks du document
      nb_chars      — longueur du chunk en caractères
      nb_words      — nombre de mots (approximatif)
      added         — timestamp ISO 8601 UTC de l'ingestion
    """
    return {
        "document_id":  document_id,
        "title":        title,
        "source":       source,
        "chunk_index":  chunk_index,
        "total_chunks": total_chunks,
        "nb_chars":     len(chunk_text),
        "nb_words":     len(chunk_text.split()),
        "added":        added,
    }


# =============================================================================
#  4. Points d'entrée publics
# =============================================================================

def ingest_text(text: str, title: str = "Document manuel", source: str = "manual") -> int:
    """Ingère une version de document, sans remplacement destructif.

    L'identité dépend du titre, de la source et de tout le texte normalisé.
    Ré-ingérer la même version est idempotent. Un changement crée une nouvelle
    version distincte et conserve l'ancienne, encore consultable en recherche.
    Aucun nettoyage des anciens IDs ni remplacement multi-processus atomique
    n'est effectué; un opérateur doit gérer explicitement les versions obsolètes.

    Returns:
        Nombre de chunks soumis (y compris les chunks déjà présents).
    """
    text = _normalize(text)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    now         = datetime.now(timezone.utc).isoformat()
    identity = json.dumps([title, source, text], ensure_ascii=False, separators=(",", ":"))
    document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    total       = len(chunks)

    metadatas = [
        _make_metadata(document_id, title, source, i, total, c, now)
        for i, c in enumerate(chunks)
    ]
    return vectorstore.add_documents(chunks, metadatas)


def ingest_file(path: str, title: str = None) -> int:
    """Ingère un fichier (TXT, MD, PDF) dans ChromaDB.

    Returns:
        Nombre de chunks indexés.
    """
    text = extract_text(path)
    if not text.strip():
        return 0
    return ingest_text(
        text,
        title  = title or os.path.basename(path),
        source = os.path.basename(path),
    )


def ingest_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    title_column: str | None = "title",
    source_column: str | None = "source",
) -> int:
    """Ingère un split Hugging Face via la dépendance optionnelle datasets.

    Les textes vides sont ignorés; les colonnes configurées absentes et les
    textes non textuels sont des erreurs explicites. Les lignes déjà indexées
    restent intactes si une ligne ultérieure échoue; une relance est idempotente
    pour des lignes inchangées. Les titres/sources absents (None) sont dérivés
    du dataset, du split et du contenu, sans dépendre de l'ordre des lignes.
    """
    if not dataset_name or not split or not text_column:
        raise ValueError("dataset_name, split and text_column must be non-empty")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face ingestion requires the optional 'datasets' package. "
            "Use an environment where it is available or ingest local files/text."
        ) from exc
    dataset = load_dataset(dataset_name, split=split)
    required = {column for column in (text_column, title_column, source_column) if column}
    available = getattr(dataset, "column_names", None)
    if available is not None and not required.issubset(available):
        raise ValueError(f"Dataset {dataset_name!r} is missing columns: {sorted(required - set(available))}")
    total = 0
    for row_index, row in enumerate(dataset):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Dataset row {row_index} is missing columns: {sorted(missing)}")
        text = row[text_column]
        if text is None:
            continue
        if not isinstance(text, str):
            raise ValueError(f"Dataset row {row_index}, column {text_column!r}, must contain text")
        if not text.strip():
            continue
        digest = hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()[:16]
        title = row.get(title_column) if title_column else None
        source = row.get(source_column) if source_column else None
        total += ingest_text(
            text,
            title=str(title) if title is not None and str(title).strip() else f"{dataset_name}:{digest}",
            source=str(source) if source is not None and str(source).strip() else f"hf://{dataset_name}/{split}",
        )
    return total


if __name__ == "__main__":
    import sys as _sys
    for p in _sys.argv[1:]:
        n = ingest_file(p)
        print(f"{p} → {n} chunks indexés")

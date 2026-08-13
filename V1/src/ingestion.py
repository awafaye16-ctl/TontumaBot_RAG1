"""Ingestion de documents : extraction texte (TXT, MD, PDF), chunking,
puis ajout à la base vectorielle via vectorstore.
"""
import os
import re
from datetime import datetime, timezone

import vectorstore

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def extract_text(path: str) -> str:
    """Extrait le texte brut d'un fichier (txt, md, pdf)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf requis pour les PDF. Installez : pip install pypdf")
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe le texte en chunks avec chevauchement, sur les limites de phrases."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > chunk_size and current:
            chunks.append(current.strip())
            tail = current[-overlap:] if len(current) > overlap else current
            current = tail + " " + s
        else:
            current = (current + " " + s).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def ingest_file(path: str, title: str = None) -> int:
    """Ingère un fichier complet dans la BD vectorielle.
    Retourne le nombre de chunks indexés."""
    text = extract_text(path)
    return ingest_text(text, title or os.path.basename(path), os.path.basename(path))


def ingest_text(text: str, title: str, source: str = "manual") -> int:
    """Ingère du texte brut dans la BD vectorielle."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    document_id = str(int(datetime.now().timestamp() * 1000))
    metas = [
        {
            "document_id": document_id,
            "title": title,
            "source": source,
            "chunk": i,
            "added": now,
        }
        for i in range(len(chunks))
    ]
    return vectorstore.add_documents(chunks, metas)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        n = ingest_file(p)
        print(f"{p}: {n} chunks indexés")
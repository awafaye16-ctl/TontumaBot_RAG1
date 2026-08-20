"""Ingestion de documents dans ChromaDB."""
import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

import vectorstore

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def ingest_text(text: str, title: str = "Document manuel") -> int:
    chunks = _chunk_text(text)
    if not chunks:
        return 0
    doc_id = hashlib.md5((title + text[:200]).encode()).hexdigest()[:12]
    metadatas = [
        {"document_id": doc_id, "title": title, "added": str(Path.home())}
        for _ in chunks
    ]
    return vectorstore.add_documents(chunks, metadatas)


def ingest_file(path: str, title: str = None) -> int:
    ext = os.path.splitext(path)[1].lower()
    title = title or os.path.basename(path)

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if not text.strip():
        return 0
    return ingest_text(text, title)

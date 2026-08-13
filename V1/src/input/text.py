"""Normalisation du texte brut (entrée directe FR/WO)."""


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def read(text: str) -> str:
    """Reçoit le texte brut et le normalise (pas d'OCR/STT)."""
    return normalize_text(text)
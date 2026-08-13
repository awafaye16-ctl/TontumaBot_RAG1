"""Speech-to-Text : audio wolof -> texte (whisper fine-tuné LoRA).

Utilise le modèle fine-tuné wolof si présent (`STT_MODEL_PATH`), sinon
retombe sur un modèle Whisper généraliste (multilingue).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import whisper

_model = None
_MODEL_SOURCE = None


def load_model():
    """Charge le modèle STT une seule fois. Priorité au LoRA wolof local."""
    global _model, _MODEL_SOURCE
    if _model is not None:
        return _model

    candidate = settings.STT_MODEL_PATH
    if candidate and os.path.isdir(candidate):
        print(f"Chargement du modèle STT wolof (LoRA) : {candidate}")
        _model = whisper.load_model(candidate)
        _MODEL_SOURCE = "wolof-lora"
    else:
        print("LoRA wolof introuvable, chargement de Whisper 'small' multilingue...")
        _model = whisper.load_model("small")
        _MODEL_SOURCE = "whisper-small"
    return _model


def transcribe(audio_path: str, language: str = None) -> str:
    """Transcrit un fichier audio en texte.
    `language` : 'wo' si le modèle le supporte, sinon auto.
    """
    model = load_model()
    kwargs = {"language": language} if language else {}
    result = model.transcribe(audio_path, **kwargs)
    return result["text"].strip()


def read(audio_path: str) -> str:
    """Point d'entrée : audio -> texte."""
    return transcribe(audio_path)
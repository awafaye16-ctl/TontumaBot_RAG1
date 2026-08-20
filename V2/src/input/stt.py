"""STT : audio -> texte via Whisper (LoRA wolof si dispo)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

import whisper

_model = None
_MODEL_SOURCE = None


def load_model():
    global _model, _MODEL_SOURCE
    if _model is not None:
        return _model
    candidate = settings.STT_MODEL_PATH
    if candidate and os.path.isdir(candidate):
        print(f"Chargement STT wolof (LoRA) : {candidate}")
        _model = whisper.load_model(candidate)
        _MODEL_SOURCE = "wolof-lora"
    else:
        print("Chargement Whisper 'small' multilingue...")
        _model = whisper.load_model("small")
        _MODEL_SOURCE = "whisper-small"
    return _model


def transcribe(audio_path: str, language: str = None) -> str:
    model = load_model()
    kwargs = {"language": language} if language else {}
    result = model.transcribe(audio_path, **kwargs)
    return result["text"].strip()

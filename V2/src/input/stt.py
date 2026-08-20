"""STT : audio -> texte via Whisper (fine-tuné wolof via transformers)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

_model = None
_processor = None
_MODEL_SOURCE = None


def load_model():
    global _model, _processor, _MODEL_SOURCE
    if _model is not None:
        return _model, _processor

    candidate = settings.STT_MODEL_PATH
    if candidate and os.path.isdir(candidate):
        print(f"Chargement STT wolof (transformers) : {candidate}")
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        _processor = WhisperProcessor.from_pretrained(candidate)
        _model = WhisperForConditionalGeneration.from_pretrained(candidate)
        _MODEL_SOURCE = "wolof-lora"
    else:
        print("Chargement Whisper 'small' multilingue (openai-whisper)...")
        import whisper
        _model = whisper.load_model("small")
        _processor = None
        _MODEL_SOURCE = "whisper-small"

    return _model, _processor


def transcribe(audio_path: str, language: str = None) -> str:
    model, processor = load_model()

    if _MODEL_SOURCE == "wolof-lora":
        import torch
        import librosa
        import numpy as np

        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    else:
        kwargs = {"language": language} if language else {}
        result = model.transcribe(audio_path, **kwargs)
        return result["text"].strip()

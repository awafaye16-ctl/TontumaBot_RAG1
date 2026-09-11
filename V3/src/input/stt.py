"""STT V3 — Transcription audio wolof.

Modèle retenu après benchmark : M9and2M/whisper-small-wolof
  → Fine-tuné spécifiquement sur le wolof, 242M paramètres.
  → Chargé depuis le dossier local `wolof-whisper-small-lora/` (déjà cloné en V3/)
    OU téléchargé depuis HuggingFace Hub si le dossier local est absent.

Le modèle est chargé une seule fois (lazy loading) et reste en mémoire.
"""
import os
import sys
import logging
from threading import RLock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings, require_model  # noqa: E402

_model     = None
_processor = None
_SOURCE    = None
_models = {}
_lock = RLock()
_log = logging.getLogger(__name__)


def resolve_language(language=None) -> str:
    value = str(language if language is not None else settings.STT_LANGUAGE).strip().lower()
    aliases = {"fr": "fr", "french": "fr", "wo": "wo", "wolof": "wo"}
    if value in {"", "auto"}:
        value = str(settings.STT_AUTO_LANGUAGE).strip().lower()
        if value not in aliases:
            raise ValueError("STT_AUTO_LANGUAGE must be fr/french or wo/wolof; auto is not language detection.")
        _log.warning("STT auto selection uses configured %s checkpoint; audio language is not detected.", aliases[value])
    if value not in aliases:
        raise ValueError("Unsupported STT language; use fr/french, wo/wolof or auto.")
    return aliases[value]


def source(language=None) -> str:
    selected = resolve_language(language)
    return settings.STT_FR_MODEL_DIR if selected == "fr" else settings.STT_WO_MODEL_DIR


def _decode_audio(audio_path):
    import av
    import numpy as np

    limit = min(30.0, settings.STT_MAX_DURATION_SECONDS)
    if limit <= 0:
        raise ValueError("STT_MAX_DURATION_SECONDS must be positive.")
    chunks = []
    samples = 0

    def append(frames):
        nonlocal samples
        for frame in frames:
            array = frame.to_ndarray().reshape(-1)
            samples += array.size
            if samples > int(limit * 16000):
                raise ValueError(f"Audio exceeds {limit:g} seconds; split it into shorter recordings. No speech was transcribed.")
            chunks.append(array)

    try:
        with av.open(str(audio_path)) as container:
            if not container.streams.audio:
                raise ValueError("Audio file contains no audio stream.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                append(resampler.resample(frame))
            append(resampler.resample(None))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Cannot decode audio: file is empty, corrupt or unsupported.") from exc
    if not samples:
        raise ValueError("Audio file is empty.")
    # Durée minimale : un enregistrement quasi vide (clic accidentel, coupure
    # immédiate du micro) peut produire une entrée dégénérée pour le modèle et
    # provoquer un crash natif au lieu d'une exception Python normale. Rejeter
    # explicitement ces cas avant d'atteindre le modèle.
    MIN_SAMPLES = int(0.3 * 16000)  # 0.3 seconde à 16 kHz
    if samples < MIN_SAMPLES:
        raise ValueError("Audio too short to transcribe (minimum 0.3 second). No speech was transcribed.")
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-7:
        raise ValueError("Audio is silent or contains invalid samples.")
    return audio

# Identifiant Hub en cas de téléchargement automatique
_HUB_MODEL_ID = "M9and2M/whisper-small-wolof"


def load_model(language=None):
    """Charge le modèle STT une seule fois.

    Priorité :
      1. Dossier local STT_MODEL_PATH (wolof-whisper-small-lora/)
      2. Téléchargement automatique depuis Hub (M9and2M/whisper-small-wolof)
    """
    global _model, _processor, _SOURCE
    selected = resolve_language(language)
    src = source(selected)
    with _lock:
        if src not in _models:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            src = require_model(src, "whisper")
            print(f"[STT] Chargement {selected.upper()} : {src}")
            processor = WhisperProcessor.from_pretrained(src)
            model = WhisperForConditionalGeneration.from_pretrained(src)
            model.eval()
            _models[src] = (model, processor)
        _model, _processor = _models[src]
        _SOURCE = src
        return _model, _processor


def transcribe(audio_path: str, language: str = None) -> str:
    """Transcrit un fichier audio en texte.

    audio_path : chemin vers le fichier audio (wav, mp3, m4a, webm…)
    language   : code langue facultatif ('wo' pour forcer le wolof)

    Utilise librosa pour le chargement audio (supporte tous les formats
    grâce à soundfile/audioread) et le processor Whisper pour les features.
    """
    selected = resolve_language(language)

    # Chargement et resample à 16 kHz (format attendu par Whisper)
    audio = _decode_audio(audio_path)
    import torch

    with _lock:
        model, processor = load_model(selected)

        # Extraction des features d'entrée
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(model.device)

        # Génération avec langue forcée si fournie
        gen_kwargs = {}
        try:
            forced_ids = processor.get_decoder_prompt_ids(
                language="french" if selected == "fr" else "wolof", task="transcribe"
            )
            gen_kwargs["forced_decoder_ids"] = forced_ids
        except (ValueError, KeyError):
            if selected != "wo":
                raise
            _log.warning("Wolof has no standard Whisper language token; using the WO checkpoint's native generation configuration.")  # si la langue n'est pas reconnue, on laisse Whisper détecter
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs["attention_mask"].to(model.device)
        with torch.inference_mode():
            predicted_ids = model.generate(input_features, **gen_kwargs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    if not transcription:
        raise ValueError("No speech could be transcribed from this audio.")
    return transcription


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage : python stt.py <audio_path> [language]")
        _sys.exit(1)
    audio = _sys.argv[1]
    lang  = _sys.argv[2] if len(_sys.argv) > 2 else None
    text  = transcribe(audio, language=lang)
    print(f"[STT] Transcription : {text}")

"""TTS : réponse vocale.

Priorité : modèle wolof SpeechT5 fine-tuné (`WOLOF_TTS_MODEL`,
bilalfaye/speecht5_tts-wolof-v0.2) qui synthétise en wolof ET en français.
Fallback : edge-tts (voix FR) si le modèle wolof est indisponible.

Pour fiabiliser SpeechT5 :
- le texte est découpé en phrases courtes (<= MAX_CHARS) et synthétisé
  segment par segment (le modèle dégénère en bruit sur les longs textes) ;
- l'audio final est normalisé (volume relevé) pour rester audible.
"""
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import edge_tts

VOICE_FR = "fr-FR-DeniseNeural"
SPEAKER_EMB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "speaker_embedding.npy")
MAX_CHARS = 80  # taille max d'un segment pour le TTS wolof
RATE = 16000

_model = None
_processor = None
_vocoder = None
_speaker_embedding = None
_model_source = None


def _load_speecht5():
    """Charge le modèle SpeechT5 wolof (une seule fois). Retourne False si échec."""
    global _model, _processor, _vocoder, _speaker_embedding, _model_source
    if _model is not None:
        return True
    checkpoint = settings.WOLOF_TTS_MODEL
    if not checkpoint:
        return False
    try:
        import torch
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Chargement du TTS wolof ({checkpoint}) sur {device.upper()}...")
        _processor = SpeechT5Processor.from_pretrained(checkpoint)
        _model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)
        _vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

        # Speaker embedding CMU Arctic (vecteur 512d, extrait localement)
        try:
            import numpy as np
            emb = np.load(SPEAKER_EMB_PATH)
            _speaker_embedding = torch.tensor(emb).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Speaker embedding indisponible, fallback aléatoire : {e}")
            _speaker_embedding = torch.randn(1, 512).to(device)

        _model_source = "speecht5-wolof"
        print("TTS wolof chargé.")
        return True
    except Exception as e:
        print(f"TTS wolof indisponible ({e}), fallback edge-tts FR.")
        _model_source = "edge-tts-fr"
        return False


def _split_text(text: str) -> list[str]:
    """Découpe le texte en segments courts, sur les frontières de phrases."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    segments = []
    for p in parts:
        while len(p) > MAX_CHARS:
            cut = p.rfind(" ", 0, MAX_CHARS)
            if cut < MAX_CHARS // 2:
                cut = MAX_CHARS
            segments.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            segments.append(p)
    return segments


def _clean_for_speech(text: str) -> str:
    """Nettoie le texte pour le TTS wolof (fragile avec chiffres/ponctuation).
    - les listes numérotées (1-, 2., 1) sont replacées par un point ;
    - chiffres isolés convertis en mots ;
    - ponctuation agressive (:, ;, -, parenthèses) remplacée par des virgules/points ;
    - mots anglais isolés retirés (le modèle ne les prononce pas)."""
    s = text
    s = re.sub(r"\b(\d+)\s*[-.)]\s*", lambda m: " " + _num_to_words(m.group(1)) + ". ", s)
    s = re.sub(r"\b(\d+)\b", lambda m: _num_to_words(m.group(1)), s)
    s = s.replace(":", ",").replace(";", ",").replace(" - ", ", ")
    s = s.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    # retirer les mots anglais isolés (le modèle ne les connaît pas)
    s = re.sub(r"\b(within|days|the|and|for|with|from|this|that|are|has|have|your|you|will|can)\b", " ", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _num_to_words(num: str) -> str:
    """Chiffre -> mot (1..20 sinon le chiffre en toutes lettres en wolof simple)."""
    words = {
        "1": "benn", "2": "ñaar", "3": "ñett", "4": "ñent", "5": "juróom",
        "6": "juróom benn", "7": "juróom ñaar", "8": "juróom ñett", "9": "juróom ñent",
        "10": "fukk", "11": "fukk ak benn", "12": "fukk ak ñaar",
        "20": "ñaar fukk", "50": "juróom fukk", "100": "teemeer",
        "1000": "junni", "5000": "juróom junni",
    }
    return words.get(num, num)


def _synth_segment(text: str) -> "torch.Tensor":
    import torch

    inputs = _processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    return _model.generate(
        inputs["input_ids"],
        speaker_embeddings=_speaker_embedding,
        vocoder=_vocoder,
        num_beams=5,
        temperature=0.6,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
    ).squeeze().detach().cpu().numpy()


def _normalize(wav) -> "numpy.ndarray":
    """Remonte le volume sans écrêter : pic cible ~0.9."""
    import numpy as np
    wav = np.asarray(wav, dtype=np.float32)
    peak = np.abs(wav).max()
    if peak < 1e-6:
        return wav
    gain = min(0.9 / peak, 6.0)  # limite à 6x pour ne pas amplifier le bruit
    return wav * gain


def synthesize(text: str, out_path: str) -> str:
    """Génère la voix. Wolof (ou FR) via SpeechT5 si dispo, sinon edge-tts FR."""
    if _load_speecht5():
        try:
            import numpy as np
            from scipy.io.wavfile import write as wavwrite

            chunks = [_synth_segment(s) for s in _split_text(_clean_for_speech(text))]
            if chunks:
                wav = np.concatenate([c for c in chunks])
                wav = _normalize(wav)
                tmp = out_path.replace(".mp3", ".wav")
                wavwrite(tmp, RATE, (wav * 32767).astype(np.int16))
                if _to_mp3(tmp, out_path):
                    os.remove(tmp)
                else:
                    os.replace(tmp, out_path)
                return out_path
        except Exception as e:
            print(f"Échec SpeechT5 ({e}), fallback edge-tts FR.")

    asyncio.run(_synth_edge(text, out_path))
    return out_path


async def _synth_edge(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, VOICE_FR)
    await communicate.save(out_path)


def _to_mp3(wav_path: str, mp3_path: str) -> bool:
    """Convertit WAV en MP3 via ffmpeg si présent."""
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k", mp3_path],
            check=True, capture_output=True,
        )
        return True
    except Exception:
        return False


def source() -> str:
    """Nom du moteur TTS effectif (pour l'affichage/trace)."""
    _load_speecht5()
    return _model_source or "edge-tts-fr"


if __name__ == "__main__":
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else "Jàmm nga fanaan. Nanga def?"
    out = sys.argv[2] if len(sys.argv) > 2 else "test_wolof.mp3"
    synthesize(txt, out)
    print(f"{source()} -> {out}")
"""TTS V2 : Oolel-Voices (voice cloning) OU SpeechT5 wolof.

Engine sélectionné via TTS_ENGINE dans .env :
  - oolel     : soynade-research/Oolel-Voices (voice cloning, meilleure qualité)
  - speecht5  : bilalfaye/speecht5_tts-wolof-v0.2 (rapide, léger)
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

import edge_tts
import torch
import soundfile as sf
import numpy as np

VOICE_FR = "fr-FR-DeniseNeural"
MAX_CHARS = 80
RATE = 16000

# ---- Oolel-Voices state ----
_oolel_model = None
_oolel_ready = False
_oolel_prompt = None

# ---- SpeechT5 state ----
_s5_model = None
_s5_processor = None
_s5_vocoder = None
_s5_speaker_emb = None
_s5_ready = False


# =========================================================================
#  Oolel-Voices — suit le même pattern que Oolel-Voices/server.py
# =========================================================================

def _get_oolel_ckpt_dir():
    """Trouve le dossier snapshot local du modèle Oolel-Voices."""
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--soynade-research--Oolel-Voices" / "snapshots"
    if cache.exists() and cache.is_dir():
        dirs = sorted(cache.iterdir())
        if dirs:
            return dirs[-1]
    return None


def _load_oolel():
    global _oolel_model, _oolel_ready, _oolel_prompt
    if _oolel_ready:
        return True
    try:
        ckpt_dir = _get_oolel_ckpt_dir()
        if ckpt_dir is None:
            print("Téléchargement Oolel-Voices...")
            from huggingface_hub import snapshot_download
            ckpt_dir = Path(snapshot_download(repo_id=settings.OOLEL_TTS_REPO))

        ckpt_str = str(ckpt_dir)
        if ckpt_str not in sys.path:
            sys.path.insert(0, ckpt_str)

        print(f"Chargement Oolel-Voices depuis {ckpt_dir}...")
        from modeling_oolel_voices import OolelVoicesForInference

        _oolel_model = OolelVoicesForInference.from_pretrained(ckpt_str, device_map="cpu")
        _oolel_model.eval()

        prompt_wav = ckpt_dir / "8_1_c.wav"
        if prompt_wav.exists():
            _oolel_prompt = str(prompt_wav)
        else:
            fallback = str(settings.BASE_DIR.parent / "Oolel-Voices" / "8_1_c.wav")
            if os.path.exists(fallback):
                _oolel_prompt = fallback

        _oolel_ready = True
        print("Oolel-Voices chargé.")
        return True
    except Exception as e:
        print(f"Oolel-Voices indisponible ({e})")
        return False


def _synth_oolel(text: str, out_path: str) -> bool:
    if not _load_oolel():
        return False
    try:
        kwargs = {}
        if _oolel_prompt:
            kwargs["audio_prompt_path"] = _oolel_prompt

        wav = _oolel_model.generate(
            text,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
            **kwargs,
        )
        audio_np = wav.squeeze(0).detach().cpu().numpy()
        sf.write(out_path, audio_np, _oolel_model.sr, format="WAV")
        return True
    except Exception as e:
        print(f"Échec Oolel-Voices ({e})")
        return False


# =========================================================================
#  SpeechT5
# =========================================================================

def _load_speecht5():
    global _s5_model, _s5_processor, _s5_vocoder, _s5_speaker_emb, _s5_ready
    if _s5_ready:
        return True
    checkpoint = settings.WOLOF_TTS_MODEL
    if not checkpoint:
        return False
    try:
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Chargement SpeechT5 ({checkpoint}) sur {device.upper()}...")
        _s5_processor = SpeechT5Processor.from_pretrained(checkpoint)
        _s5_model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)
        _s5_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        speaker_emb_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "speaker_embedding.npy")
        try:
            emb = np.load(speaker_emb_path)
            _s5_speaker_emb = torch.tensor(emb).unsqueeze(0).to(device)
        except Exception:
            _s5_speaker_emb = torch.randn(1, 512).to(device)
        _s5_ready = True
        print("SpeechT5 chargé.")
        return True
    except Exception as e:
        print(f"SpeechT5 indisponible ({e})")
        return False


def _split_text(text: str) -> list[str]:
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


def _synth_speecht5(text: str, out_path: str) -> bool:
    if not _load_speecht5():
        return False
    try:
        segments = _split_text(text)
        if not segments:
            return False
        chunks = []
        for seg in segments:
            inputs = _s5_processor(text=seg, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(_s5_model.device) for k, v in inputs.items()}
            wav = _s5_model.generate(
                inputs["input_ids"],
                speaker_embeddings=_s5_speaker_emb,
                vocoder=_s5_vocoder,
                num_beams=5,
                temperature=0.6,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
            ).squeeze().detach().cpu().numpy()
            chunks.append(wav)
        audio = np.concatenate(chunks)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * min(0.9 / peak, 6.0)
        sf.write(out_path, audio, RATE, format="WAV")
        return True
    except Exception as e:
        print(f"Échec SpeechT5 ({e})")
        return False


# =========================================================================
#  Public API
# =========================================================================

def synthesize(text: str, out_path: str, engine: str = None) -> str:
    """Génère la voix. engine : 'oolel' | 'speecht5' | None (utilise .env)."""
    engine = engine or settings.TTS_ENGINE

    if engine == "oolel":
        if _synth_oolel(text, out_path):
            return out_path

    if engine == "speecht5":
        if _synth_speecht5(text, out_path):
            return out_path

    asyncio.run(_synth_edge(text, out_path))
    return out_path


async def _synth_edge(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, VOICE_FR)
    await communicate.save(out_path)


def source(engine: str = None) -> str:
    engine = engine or settings.TTS_ENGINE
    if engine == "oolel":
        return "oolel-voices" if _oolel_ready else "edge-tts-fr"
    if engine == "speecht5":
        return "speecht5-wolof" if _s5_ready else "edge-tts-fr"
    return "edge-tts-fr"

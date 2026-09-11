"""TTS V3 — Oolel-Voices (principal) + SpeechT5 wolof (fallback) + edge-tts (dernier recours).

Architecture de fallback :
  1. Oolel-Voices  (soynade-research/Oolel-Voices) — voice cloning, meilleure qualité
     → chargé depuis HuggingFace Hub (snapshot_download) si pas en cache local
  2. SpeechT5 wolof (bilalfaye/speecht5_tts-wolof-v0.2) — si Oolel indisponible
  3. edge-tts FR  (fr-FR-DeniseNeural)               — dernier recours sans modèle

Le moteur actif est contrôlé par TTS_ENGINE dans .env :
  TTS_ENGINE=oolel      → essaie Oolel, puis SpeechT5, puis edge-tts
  TTS_ENGINE=speecht5   → essaie SpeechT5, puis edge-tts (ignore Oolel)
  TTS_ENGINE=edge       → edge-tts directement

Les modèles sont récupérés directement depuis HuggingFace Hub au premier appel.
"""
import asyncio
import os
import re
import sys
from pathlib import Path
from io import BytesIO
from functools import wraps
from threading import RLock
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings, require_model, missing_model_files, _is_local_model, _absolute_path  # noqa: E402

import edge_tts
import numpy as np
import soundfile as sf
import torch

VOICE_FR  = "fr-FR-DeniseNeural"
MAX_CHARS = 80          # taille max d'un segment SpeechT5
RATE_S5   = 16000       # sample rate SpeechT5

# ── État Oolel-Voices ──────────────────────────────────────────────────────
_oolel_model  = None
_oolel_ready  = False
_oolel_prompt = None   # chemin vers le fichier audio de référence (voice prompt)

# ── État SpeechT5 ─────────────────────────────────────────────────────────
_s5_model     = None
_s5_processor = None
_s5_vocoder   = None
_s5_spk_emb   = None
_s5_ready     = False
_lock = RLock()
_errors = {}
_last_result = ContextVar("tts_last_result", default=None)


def _synchronized(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _lock:
            return function(*args, **kwargs)
    return call


def _language(language=None):
    value = str(language or settings.TTS_LANGUAGE).strip().lower()
    aliases = {"fr": "fr", "french": "fr", "wo": "wo", "wolof": "wo"}
    if value not in aliases:
        raise ValueError("TTS language must be fr/french or wo/wolof, not auto.")
    return aliases[value]


# =============================================================================
#  Oolel-Voices
# =============================================================================

def _find_oolel_cache() -> Path | None:
    """Cherche le snapshot Oolel-Voices dans le cache HuggingFace local."""
    configured = settings.OOLEL_TTS_REPO
    explicit = any(os.getenv(key, "").strip() for key in
                   ("TTS_WO_MODEL_DIR", "OOLEL_MODEL_DIR", "OOLEL_TTS_REPO"))
    if explicit and _is_local_model(configured):
        return Path(require_model(configured, "oolel"))
    candidates = []
    if _is_local_model(configured):
        candidates.append(Path(configured))
    repo = configured if not _is_local_model(configured) else "soynade-research/Oolel-Voices"
    if not explicit:
        candidates.extend([settings.BASE_DIR / "src/models/tts_oolel_voices",
                           settings.BASE_DIR / "Oolel-Voices", settings.BASE_DIR.parent / "Oolel-Voices"])
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
    cache = Path(os.getenv("HF_HUB_CACHE", str(hf_home / "hub"))) / ("models--" + repo.replace("/", "--")) / "snapshots"
    if cache.is_dir():
        candidates.extend(sorted(cache.iterdir(), reverse=True))
    return next((path for path in candidates if not missing_model_files(path, "oolel")), None)


@_synchronized
def _load_oolel() -> bool:
    """Charge Oolel-Voices depuis le Hub (ou le cache). Retourne True si succès."""
    global _oolel_model, _oolel_ready, _oolel_prompt
    if _oolel_ready:
        return True
    try:
        ckpt_dir = _find_oolel_cache()
        if ckpt_dir is None:
            # Téléchargement depuis HuggingFace Hub
            print(f"[TTS] Téléchargement Oolel-Voices ({settings.OOLEL_TTS_REPO})...")
            from huggingface_hub import snapshot_download
            repo = settings.OOLEL_TTS_REPO
            if _is_local_model(repo):
                repo = "soynade-research/Oolel-Voices"
            ckpt_dir = Path(snapshot_download(repo_id=repo))
            require_model(str(ckpt_dir), "oolel")
            print(f"[TTS] Oolel-Voices téléchargé → {ckpt_dir}")

        # Ajoute le dossier au sys.path pour importer modeling_oolel_voices
        ckpt_str = str(ckpt_dir)
        if ckpt_str not in sys.path:
            sys.path.insert(0, ckpt_str)

        print(f"[TTS] Chargement Oolel-Voices depuis {ckpt_dir}...")
        from modeling_oolel_voices import OolelVoicesForInference  # type: ignore

        _oolel_model = OolelVoicesForInference.from_pretrained(ckpt_str, device_map="cpu")
        _oolel_model.eval()

        # Prompt audio de référence pour le voice cloning.
        # Priorité à la voix clonée "Voca" (OOLEL_VOICE_PROMPT ou Oolel-Voices/Voca.mp3),
        # puis repli sur le prompt de démo Oolel (8_1_c.wav) si présent.
        candidates = []
        if settings.OOLEL_VOICE_PROMPT:
            candidates.append(_absolute_path(settings.OOLEL_VOICE_PROMPT))
        candidates += [
            ckpt_dir / "Voca.mp3",
            settings.BASE_DIR / "Oolel-Voices" / "Voca.mp3",
            settings.BASE_DIR.parent / "Oolel-Voices" / "Voca.mp3",
            ckpt_dir / "8_1_c.wav",
            settings.BASE_DIR / "Oolel-Voices" / "8_1_c.wav",
            settings.BASE_DIR.parent / "Oolel-Voices" / "8_1_c.wav",
        ]
        for c in candidates:
            if c.exists():
                _oolel_prompt = str(c)
                print(f"[TTS] Voice prompt : {_oolel_prompt}")
                break

        if _oolel_prompt is None:
            print("[TTS] ⚠️  Aucun fichier audio de référence trouvé — synthèse sans voice prompt.")

        _oolel_ready = True
        print("[TTS] Oolel-Voices prêt.")
        return True

    except Exception as e:
        _errors["oolel"] = f"{type(e).__name__}: {e}"
        print(f"[TTS] Oolel-Voices indisponible ({e}) — passage au fallback SpeechT5.")
        return False


@_synchronized
def _synth_oolel(text: str, out_path: str) -> bool:
    """Synthétise avec Oolel-Voices. Retourne True si succès."""
    if not _load_oolel():
        return False
    try:
        kwargs = {}
        if _oolel_prompt:
            kwargs["audio_prompt_path"] = _oolel_prompt

        chunks = []
        with torch.inference_mode():
            for segment in _split_text(text):
                wav = _oolel_model.generate(
                    segment,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.8,
                    **kwargs,
                )
                audio_np = wav.detach().float().cpu().numpy().reshape(-1)
                if not audio_np.size or not np.isfinite(audio_np).all():
                    raise RuntimeError("Oolel returned empty or invalid audio.")
                chunks.append(audio_np)
        sf.write(out_path, np.concatenate(chunks), _oolel_model.sr, format="WAV", subtype="PCM_16")
        return True
    except Exception as e:
        _errors["oolel"] = f"{type(e).__name__}: {e}"
        print(f"[TTS] Oolel-Voices — échec synthèse ({e})")
        return False


# =============================================================================
#  SpeechT5 wolof (fallback)
# =============================================================================

@_synchronized
def _load_speecht5() -> bool:
    """Charge SpeechT5 depuis HuggingFace Hub. Retourne True si succès."""
    global _s5_model, _s5_processor, _s5_vocoder, _s5_spk_emb, _s5_ready
    if _s5_ready:
        return True
    checkpoint = settings.WOLOF_TTS_MODEL
    if not checkpoint:
        return False
    try:
        checkpoint = require_model(checkpoint)
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Chargement SpeechT5 ({checkpoint}) sur {device.upper()}...")
        _s5_processor = SpeechT5Processor.from_pretrained(checkpoint)
        _s5_model     = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)
        _s5_vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

        # Speaker embedding — xvector CMU Arctic depuis HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            from zipfile import ZipFile
            archive_path = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset")
            with ZipFile(archive_path) as archive:
                members = sorted(name for name in archive.namelist() if name.startswith("spkrec-xvect/") and name.endswith(".npy"))
                member = members[7306]
                if archive.getinfo(member).file_size > 16384:
                    raise ValueError("Speaker embedding file is too large.")
                emb = np.load(BytesIO(archive.read(member)), allow_pickle=False)
            _s5_spk_emb = torch.as_tensor(emb, dtype=torch.float32).reshape(1, -1).to(device)
            print("[TTS] Speaker embedding xvector chargé (CMU Arctic).")
        except Exception as e_emb:
            # Fallback : charger depuis le fichier .npy local si présent
            npy = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "speaker_embedding.npy"
            )
            if os.path.exists(npy):
                emb = np.load(npy, allow_pickle=False)
                _s5_spk_emb = torch.as_tensor(emb, dtype=torch.float32).reshape(1, -1).to(device)
                print(f"[TTS] Speaker embedding chargé depuis {npy}.")
            else:
                raise RuntimeError("SpeechT5 requires a real speaker embedding: cache spkrec-xvect.zip or provide data/speaker_embedding.npy.") from e_emb
        if _s5_spk_emb.shape != (1, 512) or not torch.isfinite(_s5_spk_emb).all():
            raise ValueError("SpeechT5 speaker embedding must contain 512 finite values.")

        _s5_model.eval()
        _s5_vocoder.eval()
        _s5_ready = True
        print("[TTS] SpeechT5 prêt.")
        return True
    except Exception as e:
        _errors["speecht5"] = f"{type(e).__name__}: {e}"
        print(f"[TTS] SpeechT5 indisponible ({e})")
        return False


def _split_text(text: str) -> list[str]:
    """Découpe le texte en segments ≤ MAX_CHARS sur les frontières de phrase."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    segments: list[str] = []
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


@_synchronized
def _synth_speecht5(text: str, out_path: str) -> bool:
    """Synthétise avec SpeechT5. Retourne True si succès."""
    if not _load_speecht5():
        return False
    try:
        segments = _split_text(text)
        if not segments:
            return False
        chunks = []
        for seg in segments:
            inputs = _s5_processor(text=seg, return_tensors="pt", padding=True, truncation=False)
            inputs = {k: v.to(_s5_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                wav = _s5_model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=_s5_spk_emb,
                    vocoder=_s5_vocoder,
                ).detach().float().cpu().numpy().reshape(-1)
            if not wav.size or not np.isfinite(wav).all():
                raise RuntimeError("SpeechT5 returned empty or invalid audio.")
            chunks.append(wav)
        audio = np.concatenate(chunks)
        # Normalisation volume (pic cible 0.9, gain max 6×)
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio * min(0.9 / peak, 6.0)
        sf.write(out_path, audio, RATE_S5, format="WAV")
        return True
    except Exception as e:
        _errors["speecht5"] = f"{type(e).__name__}: {e}"
        print(f"[TTS] SpeechT5 — échec synthèse ({e})")
        return False


# =============================================================================
#  edge-tts (dernier recours)
# =============================================================================

async def _synth_edge_async(text: str, out_path: str) -> None:
    import av

    communicate = edge_tts.Communicate(text, VOICE_FR)
    encoded = BytesIO()
    async for message in communicate.stream():
        if message["type"] == "audio":
            encoded.write(message["data"])
    if not encoded.tell():
        raise RuntimeError("Edge TTS returned no audio.")
    encoded.seek(0)
    chunks = []
    with av.open(encoded) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
        for frame in container.decode(audio=0):
            chunks.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(frame))
        chunks.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(None))
    if not chunks:
        raise RuntimeError("Edge TTS audio could not be decoded.")
    sf.write(out_path, np.concatenate(chunks), 24000, format="WAV", subtype="PCM_16")


def _synth_edge(text: str, out_path: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_synth_edge_async(text, out_path))
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(lambda: asyncio.run(_synth_edge_async(text, out_path))).result()


# =============================================================================
#  API publique
# =============================================================================

def synthesize(text: str, out_path: str, engine: str = None, language: str = None) -> str:
    """Génère la voix et sauvegarde dans out_path.

    engine : 'oolel' | 'speecht5' | 'edge' | None → utilise TTS_ENGINE du .env
    Retourne le chemin du fichier audio généré.
    """
    return synthesize_with_metadata(text, out_path, engine=engine, language=language)["path"]


@_synchronized
def synthesize_with_metadata(text: str, out_path: str, engine: str = None, language: str = None) -> dict:
    _last_result.set(None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("TTS text must not be empty.")
    if language is None and str(engine or settings.TTS_ENGINE).strip().lower() == "edge":
        language = "fr"
    language = _language(language)
    requested = str(engine or (settings.TTS_FR_ENGINE if language == "fr" else settings.TTS_WO_ENGINE)).strip().lower()
    if requested not in {"oolel", "speecht5", "edge"}:
        raise ValueError("Unknown TTS engine; use oolel, speecht5 or edge.")
    if requested == "edge" and language == "wo":
        raise ValueError("Edge's French voice cannot synthesize Wolof. Use oolel or speecht5.")
    attempts = [requested]
    if requested == "oolel":
        # Oolel a échoué → fallback SpeechT5
        attempts.append("speecht5")
    # Dernier recours : edge-tts (français)
    if language == "fr" and "edge" not in attempts:
        attempts.append("edge")
    _errors.clear()
    errors = {}
    names = {"oolel": "oolel-voices", "speecht5": "speecht5-wolof", "edge": "edge-tts-fr"}
    for actual in attempts:
        try:
            if actual == "edge":
                _synth_edge(text.strip(), str(out_path))
                success = True
            else:
                success = (_synth_oolel if actual == "oolel" else _synth_speecht5)(text.strip(), str(out_path))
            if not success:
                raise RuntimeError(_errors.get(actual, f"{actual} synthesis failed"))
            info = sf.info(str(out_path))
            if info.format != "WAV" or not info.frames:
                raise RuntimeError("TTS did not produce a non-empty WAV file.")
            result = {"path": str(out_path), "engine": names[actual], "language": language,
                      "fallback": actual != requested, "requested_engine": requested}
            if errors:
                result["errors"] = errors
                result["error"] = "; ".join(f"{name}: {message}" for name, message in errors.items())
            _last_result.set(result)
            return result
        except Exception as exc:
            errors[actual] = str(exc)
    raise RuntimeError(f"No TTS engine could synthesize {language}: " + "; ".join(
        f"{name}: {message}" for name, message in errors.items()))


def source(engine: str = None) -> str:
    """Retourne le nom du moteur TTS effectif (pour les traces)."""
    result = _last_result.get()
    if result is None or (engine and result["requested_engine"] != engine.lower()):
        return "not-synthesized"
    return result["engine"]


if __name__ == "__main__":
    import sys as _sys
    txt = _sys.argv[1] if len(_sys.argv) > 1 else "Jàmm nga fanaan. Nanga def?"
    out = _sys.argv[2] if len(_sys.argv) > 2 else "test_v3.wav"
    result = synthesize(txt, out)
    print(f"[TTS] {source()} → {result}")

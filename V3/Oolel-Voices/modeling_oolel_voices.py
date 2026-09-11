"""
Usage:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("user_id/repo_name", trust_remote_code=True)
    wav = model.generate("Hello, world!", audio_prompt_path="voice.wav")
    # wav → torch.Tensor shape (1, T) at model.sr Hz
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)

_REPO_ROOT: Optional[str] = None
_MODULES_LOADED = False

T3 = None
S3Gen = None
VoiceEncoder = None
EnTokenizer = None
T3Cond = None
S3_SR: int = 16_000
S3GEN_SR: int = 22_050
drop_invalid_tokens = None


def _set_repo_root(path: str) -> None:
    global _REPO_ROOT
    _REPO_ROOT = str(path)


def _ensure_modules_loaded() -> None:
    global _MODULES_LOADED
    global T3, S3Gen, VoiceEncoder, EnTokenizer, T3Cond
    global S3_SR, S3GEN_SR, drop_invalid_tokens

    if _MODULES_LOADED:
        return

    if _REPO_ROOT is None:
        raise RuntimeError(
            "_REPO_ROOT has not been set — from_pretrained() must call "
            "_set_repo_root() before _ensure_modules_loaded()."
        )

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    def _imp(module: str, attr: str):
        return getattr(importlib.import_module(module), attr)
    
    T3                  = _imp("models.t3.t3", "T3")
    S3Gen               = _imp("models.s3gen", "S3Gen")
    S3GEN_SR            = _imp("models.s3gen", "S3GEN_SR")
    VoiceEncoder        = _imp("models.voice_encoder", "VoiceEncoder")
    EnTokenizer         = _imp("models.tokenizers", "EnTokenizer")
    T3Cond              = _imp("models.t3.modules.cond_enc", "T3Cond")
    S3_SR               = _imp("models.s3tokenizer", "S3_SR")
    drop_invalid_tokens = _imp("models.s3tokenizer", "drop_invalid_tokens")

    _MODULES_LOADED = True


def _get_config_class():
    # When loaded by HF, __name__ is "transformers_modules.<hash>.modeling_oolel_voices"
    # We must import the config from the *same* namespace, not as a bare top-level module.
    if "." in __name__:
        parent = __name__.rsplit(".", 1)[0]
        config_module_name = f"{parent}.configuration_oolel_voices"
        try:
            mod = importlib.import_module(config_module_name)
            return getattr(mod, "OolelVoicesConfig")
        except (ImportError, AttributeError):
            pass

    # Fallback for local/notebook usage where __name__ is just the filename
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    return getattr(
        importlib.import_module("configuration_oolel_voices"),
        "OolelVoicesConfig",
    )


def _punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in [
        ("...", ", "), ("\u2026", ", "), (":", ","), (" - ", ", "),
        (";", ", "), ("\u2014", "-"), ("\u2013", "-"), (" ,", ","),
        ("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'"),
    ]:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text


@dataclass
class _Conditionals:
    t3: object
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        self.gen = {
            k: v.to(device=device) if torch.is_tensor(v) else v
            for k, v in self.gen.items()
        }
        return self

    def save(self, fpath: Path):
        torch.save(dict(t3=self.t3.__dict__, gen=self.gen), fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        _ensure_modules_loaded()
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kw = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kw["t3"]), kw["gen"])


class OolelVoicesForInference(PreTrainedModel):
    """
    Load with::

        from transformers import AutoModel
        model = AutoModel.from_pretrained("user_id/repo_name", trust_remote_code=True)
        wav = model.generate("Hello!")
    """

    # Placeholder — patched to OolelVoicesConfig at the bottom of this file.
    config_class = PretrainedConfig

    base_model_prefix = ""
    _no_split_modules = ["T3", "S3Gen", "VoiceEncoder"]

    def __init__(self, config, t3=None, s3gen=None, ve=None,
                 tokenizer=None, conds=None):
        _ensure_modules_loaded()
        super().__init__(config)
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self._conds = conds
        self.sr = config.sample_rate

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *,
                        device_map=None, torch_dtype=None,
                        trust_remote_code=False, **kwargs):
        ckpt_dir = cls._get_checkpoint_dir(pretrained_model_name_or_path)
        _set_repo_root(str(ckpt_dir))
        _ensure_modules_loaded()

        device  = cls._resolve_device(device_map)
        map_loc = torch.device("cpu") if device in ("cpu", "mps") else None

        OolelVoicesConfig = _get_config_class()
        config = OolelVoicesConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        builtin = ckpt_dir / "conds.pt"
        if builtin.exists():
            conds = _Conditionals.load(builtin, map_location=map_loc).to(device)

        model = cls(config, t3=t3, s3gen=s3gen, ve=ve,
                    tokenizer=tokenizer, conds=conds)
        model._device_name = device
        return model

    def prepare_conditionals(self, wav_fpath: str,
                              exaggeration: Optional[float] = None) -> None:
        _ensure_modules_loaded()
        exaggeration = exaggeration if exaggeration is not None \
            else self.config.exaggeration
        device = self._get_device()

        s3gen_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k = librosa.resample(s3gen_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_wav[:10 * S3GEN_SR], S3GEN_SR, device=device
        )

        toks = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            toks, _ = self.s3gen.tokenizer.forward(
                [ref_16k[:6 * S3_SR]], max_len=plen
            )
            toks = torch.atleast_2d(toks).to(device)

        ve_emb = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)
        ).mean(0, keepdim=True).to(device)

        t3_cond = T3Cond(
            speaker_emb=ve_emb,
            cond_prompt_speech_tokens=toks,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=device)

        self._conds = _Conditionals(t3_cond, s3gen_ref_dict)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        audio_prompt_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = 1024,
    ) -> torch.Tensor:
        _ensure_modules_loaded()

        exaggeration       = exaggeration       if exaggeration       is not None else self.config.exaggeration
        cfg_weight         = cfg_weight         if cfg_weight         is not None else self.config.cfg_weight
        temperature        = temperature        if temperature        is not None else self.config.temperature
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        min_p              = min_p              if min_p              is not None else self.config.min_p
        top_p              = top_p              if top_p              is not None else self.config.top_p

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

        if self._conds is None:
            raise RuntimeError(
                "No speaker conditioning. Pass audio_prompt_path or call "
                "prepare_conditionals() first."
            )

        stored = float(self._conds.t3.emotion_adv[0, 0, 0])
        if abs(exaggeration - stored) > 1e-6:
            _c = self._conds.t3
            self._conds.t3 = T3Cond(
                speaker_emb=_c.speaker_emb,
                cond_prompt_speech_tokens=_c.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self._get_device())

        device = self._get_device()
        text = _punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        speech_tokens = self.t3.inference(
            t3_cond=self._conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        speech_tokens = drop_invalid_tokens(speech_tokens[0])
        speech_tokens = speech_tokens[speech_tokens < 6561].to(device)

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self._conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        return torch.from_numpy(wav).unsqueeze(0)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use .generate() for inference; forward() is not implemented."
        )

    def _get_device(self) -> str:
        if hasattr(self, "_device_name"):
            return self._device_name
        try:
            return str(next(self.t3.parameters()).device)
        except StopIteration:
            return "cpu"

    @staticmethod
    def _resolve_device(device_map) -> str:
        if device_map in (None, "auto"):
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
            return "cpu"
        if device_map == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                logger.warning("MPS not available, falling back to CPU.")
                return "cpu"
        return device_map

    @staticmethod
    def _get_checkpoint_dir(name_or_path: str) -> Path:
        p = Path(name_or_path)
        if p.exists() and p.is_dir():
            return p
        # Check local HF cache first — avoids re-downloading when AutoModel
        # has already pulled the repo before calling our from_pretrained().
        try:
            return Path(snapshot_download(repo_id=name_or_path, local_files_only=True))
        except Exception:
            pass
        return Path(snapshot_download(repo_id=name_or_path))


# Patch config_class now that the class is defined.
# _get_config_class() uses __file__ (not _REPO_ROOT) so this works at
# import time even before from_pretrained() has been called.
OolelVoicesForInference.config_class = _get_config_class()
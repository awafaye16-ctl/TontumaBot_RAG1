"""Traduction bidirectionnelle Wolof - Français — V3.

Résultat du benchmark :
  - WO→FR : bilalfaye/nllb-200-distilled-600M-wo-fr-en
            (fine-tuné spécifiquement sur paires wolof-français-anglais,
             meilleur sur le vocabulaire administratif sénégalais)
  - FR→WO : Lahad/nllb200-francais-wolof
            (fine-tuné français→wolof, produit un wolof plus naturel
             et idiomatique que bilalfaye dans ce sens)

Les deux modèles sont chargés paresseusement (au premier appel) et
restent en mémoire pour les appels suivants.
"""
import time
import re
from functools import wraps
from threading import RLock
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Codes de langue NLLB (format : iso_Script)
WOLOF  = "wol_Latn"
FRENCH = "fra_Latn"

# ── Modèle WO→FR ──────────────────────────────────────────────────────────
# Lit depuis config (dossier local si présent, sinon Hub)
from config import settings as _settings, require_model
_WO_FR_CHECKPOINT = _settings.NLLB_WO_FR_MODEL
_FR_WO_CHECKPOINT = _settings.NLLB_FR_WO_MODEL

_tok_wo_fr   = None
_model_wo_fr = None
_tok_fr_wo   = None
_model_fr_wo = None

_device = "cuda" if torch.cuda.is_available() else "cpu"


_wo_fr_lock = RLock()
_fr_wo_lock = RLock()


def _synchronized(lock):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with lock:
                return function(*args, **kwargs)
        return call
    return decorate


@_synchronized(_wo_fr_lock)
def _load_wo_fr():
    """Charge le modèle WO→FR une seule fois."""
    global _tok_wo_fr, _model_wo_fr
    if _model_wo_fr is not None:
        return
    print(f"[NLLB] Chargement WO->FR ({_WO_FR_CHECKPOINT}) sur {_device.upper()}...")
    checkpoint = require_model(_WO_FR_CHECKPOINT, "nllb")
    tokenizer = NllbTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(_device)
    model.eval()
    _tok_wo_fr, _model_wo_fr = tokenizer, model
    print("[NLLB] WO->FR chargé.")


@_synchronized(_fr_wo_lock)
def _load_fr_wo():
    """Charge le modèle FR→WO une seule fois."""
    global _tok_fr_wo, _model_fr_wo
    if _model_fr_wo is not None:
        return
    print(f"[NLLB] Chargement FR->WO ({_FR_WO_CHECKPOINT}) sur {_device.upper()}...")
    checkpoint = require_model(_FR_WO_CHECKPOINT, "nllb")
    tokenizer = NllbTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(_device)
    model.eval()
    _tok_fr_wo, _model_fr_wo = tokenizer, model
    print("[NLLB] FR->WO chargé.")


def _text_chunks(text, tokenizer, max_tokens=384):
    def split(part):
        if len(tokenizer.encode(part, add_special_tokens=True)) <= max_tokens:
            return [part]
        midpoint = len(part) // 2
        if not midpoint:
            raise ValueError("Translation input cannot fit within the model token limit.")
        cut = part.rfind(" ", 0, midpoint + 1)
        if cut <= 0:
            cut = midpoint
        return split(part[:cut].strip()) + split(part[cut:].strip())

    chunks = []
    current = ""
    for sentence in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()):
        if not sentence.strip():
            continue
        for part in split(sentence.strip()):
            candidate = f"{current} {part}".strip()
            if current and len(tokenizer.encode(candidate, add_special_tokens=True)) > max_tokens:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _translate(text: str, tokenizer, model, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """Traduit `text` avec le modèle/tokenizer fournis."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Translation text must not be empty.")
    t0 = time.monotonic()
    tokenizer.src_lang = src_lang
    results = []
    for chunk in _text_chunks(text, tokenizer):
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=False).to(_device)
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True,
            )
        if tokens.shape[-1] >= 513:
            raise RuntimeError("Translation reached its output limit; shorten the input rather than returning a truncated answer.")
        result = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        if not result:
            raise RuntimeError("The translation model returned an empty segment.")
        results.append(result)
    return " ".join(results), round(time.monotonic() - t0, 2)


@_synchronized(_wo_fr_lock)
def wolof_to_french(text: str) -> tuple[str, float]:
    """Traduit du wolof vers le français.
    Modèle : bilalfaye/nllb-200-distilled-600M-wo-fr-en
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Translation text must not be empty.")
    _load_wo_fr()
    return _translate(text, _tok_wo_fr, _model_wo_fr, src_lang=WOLOF, tgt_lang=FRENCH)


@_synchronized(_fr_wo_lock)
def french_to_wolof(text: str) -> tuple[str, float]:
    """Traduit du français vers le wolof.
    Modèle : Lahad/nllb200-francais-wolof
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Translation text must not be empty.")
    _load_fr_wo()
    return _translate(text, _tok_fr_wo, _model_fr_wo, src_lang=FRENCH, tgt_lang=WOLOF)


if __name__ == "__main__":
    print("=== Test WO->FR ===")
    tr, d = wolof_to_french("dama beug wout kayitu juddu?")
    print(f"  WO : dama beug wout kayitu juddu?")
    print(f"  FR : {tr}  ({d}s)\n")

    print("=== Test FR->WO ===")
    tr, d = french_to_wolof("Comment obtenir un extrait de naissance ?")
    print(f"  FR : Comment obtenir un extrait de naissance ?")
    print(f"  WO : {tr}  ({d}s)")

import time
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

MODEL_NAME = "bilalfaye/nllb-200-distilled-600M-wo-fr-en"
WOLOF = "wol_Latn"
FRENCH = "fra_Latn"

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    print(f"Chargement NLLB ({MODEL_NAME}) sur {_device.upper()}...")
    _tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _model.to(_device)
    print("NLLB chargé.")


def translate(text: str, src_lang: str = WOLOF, tgt_lang: str = FRENCH) -> tuple[str, float]:
    load_model()
    t0 = time.time()
    _tokenizer.src_lang = src_lang
    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(_device)
    translated_tokens = _model.generate(
        **inputs,
        forced_bos_token_id=_tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=128,
    )
    result = _tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    elapsed = round(time.time() - t0, 2)
    return result, elapsed


def wolof_to_french(text: str) -> tuple[str, float]:
    return translate(text, src_lang=WOLOF, tgt_lang=FRENCH)


def french_to_wolof(text: str) -> tuple[str, float]:
    return translate(text, src_lang=FRENCH, tgt_lang=WOLOF)

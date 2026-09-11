"""Configuration V3 — chargée depuis .env.

Modèles sélectionnés après benchmark :
  - WO→FR : bilalfaye/nllb-200-distilled-600M-wo-fr-en
  - FR→WO : Lahad/nllb200-francais-wolof
  - STT    : M9and2M/whisper-small-wolof  (local : wolof-whisper-small-lora/)
  - TTS    : Oolel-Voices (soynade-research/Oolel-Voices)
  - LLM    : Qwen/Qwen2.5-7B-Instruct (local 4bit) ou Groq/Gemini (API)
"""
import json
import os
from pathlib import Path, PureWindowsPath

# ── Désactiver TensorFlow/Keras avant tout import ─────────────────────────
# sentence_transformers déclenche keras qui charge TF (~20s) même si on
# n'utilise pas TF. Ces variables l'empêchent de se charger.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# pas de GPU TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Force sentence_transformers à utiliser PyTorch uniquement
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # évite les warnings HF

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def _load_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()


def _payload_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return not handle.read(128).startswith(b"version https://git-lfs.github.com/spec/")


def missing_model_files(path, kind="transformers") -> list[str]:
    path = Path(path)
    if not path.is_dir():
        return ["model directory"]
    required = ["config.json"]
    if kind == "speecht5":
        required += ["spm_char.model", "preprocessor_config.json", "tokenizer_config.json"]
    if kind == "embeddings":
        required += ["modules.json"]
    if kind in {"embeddings", "reranker"}:
        required += ["tokenizer_config.json"]
        if not _payload_file(path / "tokenizer.json") and not _payload_file(path / "vocab.txt"):
            required += ["tokenizer.json"]
    if kind == "oolel":
        required += ["modeling_oolel_voices.py", "configuration_oolel_voices.py",
                     "models/__init__.py", "tokenizer.json", "ve.safetensors",
                     "t3_cfg.safetensors", "s3gen.safetensors"]
    missing = [name for name in required if not (
        (path / name).is_file() if name == "models/__init__.py" else _payload_file(path / name)
    )]
    for name in ("config.json", "tokenizer.json"):
        if name in required and name not in missing:
            try:
                valid = isinstance(json.loads((path / name).read_text(encoding="utf-8")), dict)
            except (ValueError, OSError):
                valid = False
            if not valid:
                missing.append(name)
    if kind != "oolel":
        weights_ok = any(_payload_file(path / name) for name in
                         ("model.safetensors", "pytorch_model.bin"))
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = path / name
            if index.is_file():
                try:
                    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                    weights_ok = weights_ok or (bool(shards) and all(
                        isinstance(shard, str) and not PureWindowsPath(shard).drive
                        and not PureWindowsPath(shard).root and not Path(shard).is_absolute()
                        and ".." not in PureWindowsPath(shard).parts
                        and _payload_file(path / shard) for shard in shards))
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    pass
        if not weights_ok:
            missing.append("complete PyTorch/safetensors weights (including all indexed shards)")
    if kind == "whisper":
        if not _payload_file(path / "preprocessor_config.json"):
            missing.append("preprocessor_config.json")
        for name in ("vocab.json", "merges.txt"):
            if not _payload_file(path / name):
                missing.append(name)
    if kind == "nllb" and not _payload_file(path / "sentencepiece.bpe.model"):
        missing.append("sentencepiece.bpe.model")
    return missing


def _absolute_path(value) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else BASE_DIR / path).resolve()


def _is_local_model(value: str) -> bool:
    return (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
            or value.startswith((".", "~", "src/", "src\\")) or "\\" in value
            or value.count("/") != 1 or (BASE_DIR / value).exists())


def model_status(source: str, kind="transformers") -> dict:
    result = {"path": source, "type": "hub", "state": "not_cached", "exists": False,
              "files_available": False, "inference_tested": False, "remote_available": None,
              "missing": [], "validation": "required_files_only"}
    try:
        if _is_local_model(source):
            path = _absolute_path(source)
            result["type"] = "local"
        else:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(source, "config.json", revision="main")
            if not isinstance(cached, str):
                result["missing"] = ["main revision config.json not cached"]
                return result
            path = Path(cached).parent
            result["type"] = "cache_hf"
        missing = missing_model_files(path, kind)
        result.update(path=str(path), exists=path.is_dir(), missing=missing,
                      files_available=not missing, state="incomplete" if missing else "files_present")
    except (OSError, ValueError, ImportError) as exc:
        result.update(state="check_failed", missing=[type(exc).__name__])
    return result


def configured_models() -> dict:
    entries = {
        "stt_wo": (settings.STT_WO_MODEL_DIR, "whisper"),
        "stt_fr": (settings.STT_FR_MODEL_DIR, "whisper"),
        "nllb_wo_fr": (settings.NLLB_WO_FR_MODEL, "nllb"),
        "nllb_fr_wo": (settings.NLLB_FR_WO_MODEL, "nllb"),
        "tts_oolel": (settings.OOLEL_TTS_REPO, "oolel"),
        "tts_speecht5": (settings.WOLOF_TTS_MODEL, "speecht5"),
        "tts_vocoder": ("microsoft/speecht5_hifigan", "transformers"),
        "embeddings": (settings.EMBED_MODEL, "embeddings"),
        "reranker": (settings.RERANKER_MODEL, "reranker"),
    }
    if settings.LLM_PROVIDER == "local":
        entries["llm_local"] = (settings.LOCAL_LLM_MODEL, "transformers")
    return entries


def require_model(source: str, kind="transformers") -> str:
    if _is_local_model(source):
        path = _absolute_path(source)
        missing = missing_model_files(path, kind)
        if missing:
            raise FileNotFoundError(f"Incomplete model directory {path}: {', '.join(missing)}. Run download_models.py or correct the model setting.")
        return str(path)
    return source


def _resolve_model(keys, legacy, candidates, hub, kind="transformers") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return str(_absolute_path(value))
    for key in legacy:
        value = os.getenv(key, "").strip()
        if value:
            if _is_local_model(value):
                return str(_absolute_path(value))
            if value != hub:
                return value
            break
    paths = [_absolute_path(path) for path in candidates]
    for path in paths:
        if not missing_model_files(path, kind):
            return str(path)
    for path in paths:
        if path.is_dir():
            return str(path)
    return hub


class Settings:
    BASE_DIR = BASE_DIR

    MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "2"))
    # SESSION_TTL_SECONDS est un alias accepté (nom utilisé côté Borne_V1/Redis).
    SESSION_INACTIVITY_SECONDS = int(os.getenv("SESSION_INACTIVITY_SECONDS", os.getenv("SESSION_TTL_SECONDS", "900")))
    # Nombre de tours (question+réponse) conservés dans le contexte de reformulation.
    MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "5"))

    # ── LLM ──────────────────────────────────────────────────────────────
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
    LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq")
    GROQ_MODEL      = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "4096"))
    HF_DOCUMENTS_DATASET = os.getenv("HF_DOCUMENTS_DATASET", "")
    HF_DATASET_SPLIT = HF_DOCUMENTS_SPLIT = os.getenv("HF_DATASET_SPLIT", os.getenv("HF_DOCUMENTS_SPLIT", "train"))
    HF_TEXT_COLUMN = HF_DOCUMENTS_TEXT_COLUMN = os.getenv("HF_TEXT_COLUMN", os.getenv("HF_DOCUMENTS_TEXT_COLUMN", "text"))
    HF_TITLE_COLUMN = HF_DOCUMENTS_TITLE_COLUMN = os.getenv("HF_TITLE_COLUMN", os.getenv("HF_DOCUMENTS_TITLE_COLUMN", "title"))
    HF_SOURCE_COLUMN = HF_DOCUMENTS_SOURCE_COLUMN = os.getenv("HF_SOURCE_COLUMN", os.getenv("HF_DOCUMENTS_SOURCE_COLUMN", "source"))
    SEED_ENABLED = os.getenv("SEED_ENABLED", "false").lower() in {"1", "true", "yes"}
    PRELOAD_TTS = os.getenv("PRELOAD_TTS", "false").lower() in {"1", "true", "yes"}
    # Précharge STT (FR+WO), embeddings, reranker et NLLB (WO<->FR) au démarrage du
    # serveur plutôt qu'à la 1ère requête : évite qu'une première requête audio/RAG
    # paie un chargement à froid de plusieurs modèles lourds (jusqu'à 60-90s sur CPU).
    PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in {"1", "true", "yes"}
    # Modèle local (utilisé si LLM_PROVIDER == "local")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LOCAL_LLM_QUANT = os.getenv("LOCAL_LLM_QUANT", "4bit")  # 4bit | 8bit | fp16

    # ── Traduction — deux modèles distincts (résultat benchmark) ─────────
    # WO→FR : bilalfaye fine-tuné wolof-français-anglais
    NLLB_WO_FR_MODEL = _resolve_model(
        ("WO_FR_MODEL_DIR", "NLLB_WO_FR_MODEL_DIR"), ("NLLB_WO_FR_MODEL", "NLLB_WO_FR_MODEL_PATH"),
        ("src/models/wo_fr_nllb", "src/Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en"),
        "bilalfaye/nllb-200-distilled-600M-wo-fr-en", "nllb"
    )
    WO_FR_MODEL_DIR = NLLB_WO_FR_MODEL_DIR = NLLB_WO_FR_MODEL
    # FR→WO : Lahad fine-tuné français→wolof
    NLLB_FR_WO_MODEL = _resolve_model(
        ("FR_WO_MODEL_DIR", "NLLB_FR_WO_MODEL_DIR"), ("NLLB_FR_WO_MODEL", "NLLB_FR_WO_MODEL_PATH"),
        ("src/models/fr_wo_nllb", "src/Fr-Wo_Lahadnllb200-francais-wolof"),
        "Lahad/nllb200-francais-wolof", "nllb"
    )
    FR_WO_MODEL_DIR = NLLB_FR_WO_MODEL_DIR = NLLB_FR_WO_MODEL

    # ── STT ───────────────────────────────────────────────────────────────
    # Priorité : dossier local src/stt_wolof-whisper-small-lora/
    #            puis wolof-whisper-small-lora/ (déjà cloné en V3/)
    #            puis Hub M9and2M/whisper-small-wolof

    @staticmethod
    def _stt_local_valid(path: "Path") -> bool:
        """Vérifie que le dossier contient config.json ET les poids du modèle."""
        # Au moins un fichier de poids doit être présent
        return not missing_model_files(path, "whisper")

    @staticmethod
    def _resolve_stt_path() -> str:
        return _resolve_model(
            ("STT_WO_MODEL_DIR",), ("STT_WO_MODEL_PATH", "STT_MODEL_PATH"),
            ("src/models/stt_wolof_whisper_small", "src/stt_wolof-whisper-small-lora", "wolof-whisper-small-lora"),
            "M9and2M/whisper-small-wolof", "whisper"
        )  # téléchargement Hub

    STT_WO_MODEL_DIR = _resolve_stt_path.__func__()
    STT_MODEL_PATH = STT_WO_MODEL_PATH = STT_WO_MODEL_DIR
    STT_FR_MODEL_DIR = _resolve_model(
        ("STT_FR_MODEL_DIR",), ("STT_FR_MODEL_PATH",),
        ("src/models/stt_french_whisper_medium", "src/stt_french_whisper_medium", "whisper-medium-french"),
        "pierreguillou/whisper-medium-french", "whisper"
    )
    STT_FR_MODEL_PATH = STT_FR_MODEL_DIR
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")
    STT_AUTO_LANGUAGE = os.getenv("STT_AUTO_LANGUAGE", "wo")
    STT_MAX_DURATION_SECONDS = min(30.0, float(os.getenv("STT_MAX_DURATION_SECONDS", "30")))

    # ── Embeddings / Vectorstore ──────────────────────────────────────────
    EMBED_MODEL = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ── TTS ───────────────────────────────────────────────────────────────
    # Moteur par défaut : oolel (meilleur qualité selon benchmark)
    TTS_ENGINE       = os.getenv("TTS_ENGINE", "oolel")
    OOLEL_TTS_REPO = _resolve_model(
        ("TTS_WO_MODEL_DIR", "OOLEL_MODEL_DIR"), ("OOLEL_TTS_REPO",),
        ("src/models/tts_oolel_voices", "Oolel-Voices", "../Oolel-Voices"),
        "soynade-research/Oolel-Voices", "oolel"
    )
    TTS_WO_MODEL_DIR = OOLEL_MODEL_DIR = OOLEL_TTS_REPO
    # Voix clonée de référence pour Oolel-Voices (voice cloning). Prioritaire sur
    # les candidats par défaut (Voca.mp3, puis 8_1_c.wav) si renseignée.
    OOLEL_VOICE_PROMPT = os.getenv("OOLEL_VOICE_PROMPT", "").strip()
    TTS_FR_ENGINE = os.getenv("TTS_FR_ENGINE", "edge")
    TTS_WO_ENGINE = os.getenv("TTS_WO_ENGINE", TTS_ENGINE)
    TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "wo")
    # Fallback SpeechT5 si Oolel indisponible
    WOLOF_TTS_MODEL = _resolve_model(
        ("SPEECHT5_MODEL_DIR",), ("WOLOF_TTS_MODEL",), (),
        "bilalfaye/speecht5_tts-wolof-v0.2"
    )

    # ── Reranker ──────────────────────────────────────────────────────────
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
    _min = os.getenv("RERANKER_MIN_SCORE", "")
    RERANKER_MIN_SCORE = float(_min) if _min else None

    # ── Serveur ───────────────────────────────────────────────────────────
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    @property
    def llm_ready(self) -> bool:
        if self.LLM_PROVIDER == "local":
            return model_status(self.LOCAL_LLM_MODEL)["files_available"]  # toujours disponible si les poids sont téléchargés
        if self.LLM_PROVIDER == "gemini":
            return bool(self.GEMINI_API_KEY)
        return self.LLM_PROVIDER == "groq" and bool(self.GROQ_API_KEY)


settings = Settings()

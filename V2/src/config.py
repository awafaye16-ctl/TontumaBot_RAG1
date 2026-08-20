"""Configuration V2 — chargée depuis .env."""
import os
from pathlib import Path

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
        os.environ.setdefault(key.strip(), value.strip())


_load_env()


class Settings:
    BASE_DIR = BASE_DIR
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    NLLB_MODEL = os.getenv("NLLB_MODEL", "bilalfaye/nllb-200-distilled-600M-wo-fr-en")
    STT_MODEL_PATH = os.getenv("STT_MODEL_PATH", "./wolof-whisper-small-lora")
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    TTS_ENGINE = os.getenv("TTS_ENGINE", "speecht5")
    WOLOF_TTS_MODEL = os.getenv("WOLOF_TTS_MODEL", "bilalfaye/speecht5_tts-wolof-v0.2")
    OOLEL_TTS_REPO = os.getenv("OOLEL_TTS_REPO", "soynade-research/Oolel-Voices")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    @property
    def llm_ready(self) -> bool:
        if self.LLM_PROVIDER == "gemini":
            return bool(self.GEMINI_API_KEY)
        return bool(self.GROQ_API_KEY)


settings = Settings()

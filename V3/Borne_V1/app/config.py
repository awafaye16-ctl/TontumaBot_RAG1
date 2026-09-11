"""Configuration centralisée via Pydantic Settings v2.

Les valeurs proviennent :
  1. des variables d'environnement ;
  2. d'un fichier .env chargé par python-dotenv.

Aucun secret n'est écrit en dur dans le code.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    redis_url: str = Field(default="redis://redis:6379/0")
    session_ttl_seconds: int = Field(default=1800, ge=1)
    max_messages_per_session: int = Field(default=20, ge=1)
    api_key: str = Field(default="change-me")

    # Limite de longueur du texte utilisateur (caractères).
    max_text_length: int = Field(default=2000, ge=1)


@lru_cache
def get_settings() -> Settings:
    """Retourne une instance unique de Settings (cache process)."""
    return Settings()

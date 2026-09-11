"""Dépendances FastAPI : client Redis partagé et validation de la clé API.

Le client Redis est créé une seule fois par process et fermé à l'arrêt.
La clé API est validée via l'en-tête `X-API-Key` (ou `Authorization: Bearer`).
"""
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import Depends, Header, HTTPException, status

from app.config import get_settings
from app.services.session_service import SessionService

_redis: aioredis.Redis | None = None
_session_service: SessionService | None = None


async def get_redis() -> AsyncIterator[aioredis.Redis]:
    """Retourne le client Redis partagé (singleton process)."""
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    yield _redis


async def get_session_service(
    redis: aioredis.Redis = Depends(get_redis),
) -> AsyncIterator[SessionService]:
    """Retourne le service de session partagé (singleton process)."""
    global _session_service
    if _session_service is None or _session_service._redis is not redis:
        _session_service = SessionService(redis)
    yield _session_service


async def verify_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    """Valide la clé API passée en `X-API-Key` ou `Authorization: Bearer <key>`."""
    settings = get_settings()
    provided = x_api_key
    if provided is None and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1]
    if provided is None or provided != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante ou invalide.",
        )
    return provided


async def close_redis() -> None:
    """À appeler lors de l'arrêt de l'application."""
    global _redis, _session_service
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        _session_service = None

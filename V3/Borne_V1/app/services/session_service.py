"""Gestion asynchrone des sessions conversationnelles dans Redis.

Une session est stockée sous la clé `session:{session_id}` et sérialisée en JSON.
Le TTL Redis est fixé à `SESSION_TTL_SECONDS` (1800 s) à la création et renouvelé
après chaque question valide. Une session ayant atteint `max_messages` est
supprimée automatiquement.

Aucune erreur Redis n'est masquée : elle remonte sous forme de `SessionError`
pour être traduite en HTTP 503 par la couche routes.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings
from app.schemas import HistoryMessage

_log = logging.getLogger(__name__)


class SessionError(Exception):
    """Erreur métier liée à la session (Redis, état invalide, etc.)."""


class SessionNotFound(Exception):
    """Session inexistante ou expirée."""


class SessionLimitReached(Exception):
    """Limite de 20 questions atteinte."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_payload(max_messages: int) -> dict[str, Any]:
    now = _now_iso()
    return {
        "session_id": str(uuid.uuid4()),
        "status": "active",
        "messages": [],
        "messages_used": 0,
        "max_messages": max_messages,
        "created_at": now,
        "last_activity": now,
    }


class SessionService:
    def __init__(self, redis: aioredis.Redis):
        self._redis = redis
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Helpers Redis
    # ------------------------------------------------------------------
    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    async def _get_raw(self, session_id: str) -> dict[str, Any]:
        try:
            raw = await self._redis.get(self._key(session_id))
        except aioredis.RedisError as exc:  # type: ignore[attr-defined]
            raise SessionError(f"Redis indisponible : {exc.__class__.__name__}") from exc
        if not raw:
            raise SessionNotFound(session_id)
        return json.loads(raw)

    async def _save(self, payload: dict[str, Any], ttl: int | None = None) -> int:
        ttl = ttl if ttl is not None else self._settings.session_ttl_seconds
        data = json.dumps(payload, ensure_ascii=False)
        try:
            await self._redis.set(self._key(payload["session_id"]), data, ex=ttl)
        except aioredis.RedisError as exc:  # type: ignore[attr-defined]
            raise SessionError(f"Redis indisponible : {exc.__class__.__name__}") from exc
        return ttl

    async def _delete(self, session_id: str) -> None:
        try:
            await self._redis.delete(self._key(session_id))
        except aioredis.RedisError as exc:  # type: ignore[attr-defined]
            raise SessionError(f"Redis indisponible : {exc.__class__.__name__}") from exc

    async def _ttl(self, session_id: str) -> int:
        try:
            return int(await self._redis.ttl(self._key(session_id)))
        except aioredis.RedisError as exc:  # type: ignore[attr-defined]
            raise SessionError(f"Redis indisponible : {exc.__class__.__name__}") from exc

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    async def start(self) -> dict[str, Any]:
        payload = _new_session_payload(self._settings.max_messages_per_session)
        ttl = await self._save(payload)
        _log.info("Session créée (id tronqué=%s…)", payload["session_id"][:8])
        return {**payload, "ttl_seconds": ttl}

    async def status(self, session_id: str) -> dict[str, Any]:
        try:
            payload = await self._get_raw(session_id)
        except SessionNotFound:
            return {
                "session_id": session_id,
                "status": "expired",
                "messages_used": 0,
                "remaining_messages": 0,
                "ttl_seconds": 0,
            }
        ttl = await self._ttl(session_id)
        used = payload["messages_used"]
        return {
            "session_id": session_id,
            "status": payload["status"],
            "messages_used": used,
            "remaining_messages": max(0, payload["max_messages"] - used),
            "ttl_seconds": ttl,
        }

    async def add_message(self, session_id: str, text: str, language: str,
                          input_type: str = "text") -> dict[str, Any]:
        payload = await self._get_raw(session_id)
        used = payload["messages_used"]
        if used >= payload["max_messages"]:
            # Nettoyage immédiat : la 21e question est refusée.
            await self._delete(session_id)
            raise SessionLimitReached(session_id)

        # 1) On enregistre la question utilisateur.
        payload["messages"].append(
            HistoryMessage(role="user", content=text, language=language,
                           timestamp=datetime.now(timezone.utc)).model_dump(mode="json")
        )

        # 2) On appelle le pipeline RAG (fictif pour l'instant).
        rag = await process_with_rag(text, language, payload["messages"])
        answer_text = rag["answer"]
        answer_lang = rag["language"]

        # 3) On enregistre la réponse assistant.
        payload["messages"].append(
            HistoryMessage(role="assistant", content=answer_text, language=answer_lang,
                           timestamp=datetime.now(timezone.utc)).model_dump(mode="json")
        )

        payload["messages_used"] = used + 1
        payload["last_activity"] = _now_iso()

        # Si on atteint le quota, on supprime après avoir renvoyé la réponse.
        reached_limit = payload["messages_used"] >= payload["max_messages"]
        ttl = await self._save(payload, ttl=self._settings.session_ttl_seconds)
        if reached_limit:
            await self._delete(session_id)

        return {
            "session_id": session_id,
            "answer": answer_text,
            "messages_used": payload["messages_used"],
            "remaining_messages": max(0, payload["max_messages"] - payload["messages_used"]),
            "ttl_seconds": ttl,
            "status": "active" if not reached_limit else "closed",
        }

    async def end(self, session_id: str) -> None:
        # On tente de supprimer même si la session a déjà expiré.
        try:
            await self._get_raw(session_id)
        except SessionNotFound:
            raise
        except SessionError:
            raise
        await self._delete(session_id)


# ---------------------------------------------------------------------------
# Pipeline RAG fictif (à remplacer par le vrai pipeline TONTOUMA-BOT).
# ---------------------------------------------------------------------------
async def process_with_rag(text: str, language: str,
                           history: list[dict]) -> dict[str, Any]:
    """Réponse fictive. Signature stable pour brancher le vrai pipeline plus tard.

    Retourne :
      - answer : texte de la réponse
      - language : langue de la réponse
      - sources : liste des sources utilisées
      - confidence : score de confiance [0, 1]
    """
    return {
        "answer": f"[RAG fictif] Réponse à : {text}",
        "language": language,
        "sources": [],
        "confidence": 0.0,
    }

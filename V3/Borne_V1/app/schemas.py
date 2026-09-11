"""Schémas Pydantic v2 partagés entre routes, service et tests.

Toutes les réponses API passent par ces modèles : cela garantit
une structure JSON stable et documentée par OpenAPI.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Requêtes
# ---------------------------------------------------------------------------

class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Question utilisateur (non vide, <= 2000 caractères).")
    language: str = Field(default="fr", description="Langue de la question (fr/wo).")
    input_type: Literal["text", "audio", "image"] = Field(
        default="text", description="Type d'entrée utilisateur."
    )

    @field_validator("text")
    @classmethod
    def _non_empty_stripped(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le texte ne peut pas être vide.")
        return v


# ---------------------------------------------------------------------------
# Réponses
# ---------------------------------------------------------------------------

class SessionStartResponse(BaseModel):
    session_id: str
    status: Literal["active"] = "active"
    messages_used: int = 0
    remaining_messages: int
    ttl_seconds: int


class SessionStatusResponse(BaseModel):
    session_id: str
    status: Literal["active", "expired"]
    messages_used: int
    remaining_messages: int
    ttl_seconds: int


class MessageResponse(BaseModel):
    session_id: str
    answer: str
    messages_used: int
    remaining_messages: int
    ttl_seconds: int
    status: Literal["active", "closed"] = "active"


class SessionEndResponse(BaseModel):
    session_id: str
    status: Literal["closed"] = "closed"
    message: str


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Historique
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    language: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Pipeline RAG (signature future)
# ---------------------------------------------------------------------------

class RagAnswer(BaseModel):
    answer: str
    language: str
    sources: list[dict] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

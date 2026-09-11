"""Endpoints de gestion des sessions conversationnelles.

Règles métier :
  - 404 si la session n'existe pas (ou a expiré).
  - 503 si Redis est indisponible.
  - 409 si la limite de 20 questions est atteinte.
  - 422 si le texte est vide ou trop long (validation Pydantic).
  - 401 si la clé API est absente ou invalide.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_session_service, verify_api_key
from app.schemas import (
    ErrorResponse,
    MessageRequest,
    MessageResponse,
    SessionEndResponse,
    SessionStartResponse,
    SessionStatusResponse,
)
from app.services.session_service import (
    SessionError,
    SessionLimitReached,
    SessionNotFound,
    SessionService,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"],
                   dependencies=[Depends(verify_api_key)],
                   responses={401: {"model": ErrorResponse}})


def _remaining(used: int, max_messages: int) -> int:
    return max(0, max_messages - used)


@router.post("/start", response_model=SessionStartResponse,
             status_code=status.HTTP_201_CREATED)
async def start_session(
    svc: SessionService = Depends(get_session_service),
):
    try:
        payload = await svc.start()
    except SessionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return SessionStartResponse(
        session_id=payload["session_id"],
        messages_used=payload["messages_used"],
        remaining_messages=_remaining(payload["messages_used"], payload["max_messages"]),
        ttl_seconds=payload["ttl_seconds"],
    )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(
    session_id: str,
    svc: SessionService = Depends(get_session_service),
):
    try:
        data = await svc.status(session_id)
    except SessionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    # `status` renvoie "expired" si la session n'existe plus : on renvoie 404.
    if data["status"] == "expired":
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "Session inexistante ou expirée.")
    return SessionStatusResponse(**data)


@router.post("/{session_id}/message", response_model=MessageResponse)
async def send_message(
    session_id: str,
    req: MessageRequest,
    svc: SessionService = Depends(get_session_service),
):
    try:
        data = await svc.add_message(session_id, req.text, req.language, req.input_type)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "Session inexistante ou expirée.") from exc
    except SessionLimitReached as exc:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Limite de 20 questions atteinte.") from exc
    except SessionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return MessageResponse(**data)


@router.post("/{session_id}/end", response_model=SessionEndResponse)
async def end_session(
    session_id: str,
    svc: SessionService = Depends(get_session_service),
):
    try:
        await svc.end(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "Session inexistante ou expirée.") from exc
    except SessionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return SessionEndResponse(
        session_id=session_id,
        message="La session est terminée.",
    )

"""Tests pytest pour les endpoints de gestion de sessions.

Utilise httpx.AsyncClient + ASGITransport pour tester l'app en mémoire,
avec un faux Redis (FakeRedis) pour éviter une dépendance externe.
"""
import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from redis.exceptions import RedisError

from app.config import get_settings
from app.dependencies import get_redis, get_session_service
from app.main import app
from app.services.session_service import SessionService


# ---------------------------------------------------------------------------
# Faux Redis asynchrone (en mémoire) — suffisant pour les tests unitaires.
# ---------------------------------------------------------------------------
class FakeRedis:
    def __init__(self):
        self._data: dict[str, str] = {}
        self._ttl: dict[str, int] = {}
        self._broken = False

    async def get(self, key: str) -> str | None:
        if self._broken:
            raise RedisError("redis broken")
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> str:
        if self._broken:
            raise RedisError("redis broken")
        self._data[key] = value
        self._ttl[key] = ex or 0
        return "OK"

    async def delete(self, key: str) -> int:
        if self._broken:
            raise RedisError("redis broken")
        existed = key in self._data
        self._data.pop(key, None)
        self._ttl.pop(key, None)
        return 1 if existed else 0

    async def ttl(self, key: str) -> int:
        if self._broken:
            raise RedisError("redis broken")
        return self._ttl.get(key, -2)

    async def aclose(self) -> None:
        pass


@pytest_asyncio.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest_asyncio.fixture
def settings():
    return get_settings()


@pytest_asyncio.fixture
def session_service(fake_redis: FakeRedis, settings) -> SessionService:
    return SessionService(fake_redis)


# ---------------------------------------------------------------------------
# Override des dépendances FastAPI.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client(fake_redis: FakeRedis, session_service: SessionService) -> AsyncClient:
    async def _get_redis():
        yield fake_redis

    async def _get_session_service():
        yield session_service

    app.dependency_overrides[get_redis] = _get_redis
    app.dependency_overrides[get_session_service] = _get_session_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


API_KEY = "change-me"
HEADERS = {"X-API-Key": API_KEY}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_start_session_returns_20_remaining(client: AsyncClient):
    resp = await client.post("/api/sessions/start", headers=HEADERS)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "active"
    assert data["messages_used"] == 0
    assert data["remaining_messages"] == 20
    assert data["ttl_seconds"] == 1800
    assert "session_id" in data


@pytest.mark.asyncio
async def test_send_message_decrements_counter(client: AsyncClient):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]

    msg = await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "Où est la radiologie ?", "language": "fr", "input_type": "text"},
    )
    assert msg.status_code == 200, msg.text
    data = msg.json()
    assert data["messages_used"] == 1
    assert data["remaining_messages"] == 19
    assert data["status"] == "active"


@pytest.mark.asyncio
async def test_ttl_renewed_after_message(client: AsyncClient, fake_redis: FakeRedis):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    # Réduit artificiellement le TTL avant l'envoi du message.
    fake_redis._ttl[f"session:{sid}"] = 100
    await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "Bonjour", "language": "fr"},
    )
    assert fake_redis._ttl[f"session:{sid}"] == 1800


@pytest.mark.asyncio
async def test_end_session_deletes_key(client: AsyncClient, fake_redis: FakeRedis):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    end = await client.post(f"/api/sessions/{sid}/end", headers=HEADERS)
    assert end.status_code == 200
    assert end.json()["status"] == "closed"
    assert fake_redis._data.get(f"session:{sid}") is None


@pytest.mark.asyncio
async def test_unknown_session_returns_404(client: AsyncClient):
    resp = await client.get("/api/sessions/unknown-id/status", headers=HEADERS)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_message_after_expiration_returns_404(client: AsyncClient, fake_redis: FakeRedis):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    # Simule l'expiration Redis.
    del fake_redis._data[f"session:{sid}"]
    resp = await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "Encore ?", "language": "fr"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_21st_message_refused(client: AsyncClient, fake_redis: FakeRedis):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    # Force le compteur à 19, puis envoie 2 messages.
    raw = fake_redis._data[f"session:{sid}"]
    payload = json.loads(raw)
    payload["messages_used"] = 19
    fake_redis._data[f"session:{sid}"] = json.dumps(payload)

    m1 = await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "Q20", "language": "fr"},
    )
    assert m1.status_code == 200
    assert m1.json()["remaining_messages"] == 0
    assert m1.json()["status"] == "closed"

    # La 21e doit être refusée (session supprimée).
    m2 = await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "Q21", "language": "fr"},
    )
    assert m2.status_code == 404


@pytest.mark.asyncio
async def test_empty_text_rejected(client: AsyncClient):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    resp = await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "   ", "language": "fr"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_redis_error_returns_503(client: AsyncClient, fake_redis: FakeRedis):
    fake_redis._broken = True
    resp = await client.post("/api/sessions/start", headers=HEADERS)
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_response_json_structure(client: AsyncClient):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    keys = set(start.json().keys())
    assert keys == {"session_id", "status", "messages_used", "remaining_messages", "ttl_seconds"}

    msg = await client.post(
        f"/api/sessions/{sid}/message",
        headers=HEADERS,
        json={"text": "Test structure", "language": "fr"},
    )
    keys = set(msg.json().keys())
    assert keys == {"session_id", "answer", "messages_used", "remaining_messages", "ttl_seconds", "status"}


@pytest.mark.asyncio
async def test_missing_api_key_returns_401(client: AsyncClient):
    resp = await client.post("/api/sessions/start")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_status_returns_active_state(client: AsyncClient):
    start = await client.post("/api/sessions/start", headers=HEADERS)
    sid = start.json()["session_id"]
    resp = await client.get(f"/api/sessions/{sid}/status", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"

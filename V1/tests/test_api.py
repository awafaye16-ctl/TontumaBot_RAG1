"""Tests des endpoints FastAPI (client de test, sans serveur)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["llm_provider"] in ("groq", "gemini")


def test_index_serve_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "TontumaBot" in r.text


def test_ask_question_fr():
    r = client.post("/ask", json={"question": "Comment créer une entreprise au Sénégal ?"})
    assert r.status_code == 200
    data = r.json()
    assert data["response"]
    assert data["trace"]["intent"] == "procedure"


def test_ask_question_vide_erreur_400():
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 400


def test_ask_orientation():
    r = client.post("/ask", json={"question": "Où dois-je déposer ma demande de passeport ?"})
    assert r.status_code == 200
    assert r.json()["trace"]["intent"] == "orientation"
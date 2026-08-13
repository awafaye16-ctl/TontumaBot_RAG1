"""Tests du pipeline sans clé API (mode fallback) ni modèles lourds."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import build_retrievers, answer_legacy

# Force le mode fallback (pas de clé API)
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

DOCS = [
    "Pour obtenir une carte d'identité nationale, présentez une copie de l'extrait de naissance et deux photos au guichet de la mairie.",
    "Créer une entreprise au Sénégal nécessite la carte d'identité et un extrait du registre de commerce.",
]

FILTERED = [
    {"text": "La demande d'extrait de naissance se fait à la mairie.", "meta": {"lieu": True}},
]


def _retrievers():
    return build_retrievers(DOCS, FILTERED)


def test_answer_francais_procedure():
    hybrid, filtered = _retrievers()
    r = answer_legacy("Comment obtenir une carte d'identité ?", hybrid, filtered)
    assert r["trace"]["input_lang"] == "fr"
    assert r["trace"]["intent"] == "procedure"
    assert r["trace"]["search"] in ("hybrid", "hybride (seed)")
    assert r["response"], "La réponse ne doit pas être vide"


def test_answer_francais_orientation():
    hybrid, filtered = _retrievers()
    r = answer_legacy("Où dois-je déposer ma demande ?", hybrid, filtered)
    assert r["trace"]["intent"] == "orientation"
    assert r["trace"]["search"] in ("filtered", "filtrée (seed)")
    assert r["response"]


def test_answer_wolof_trace_translation():
    # Sans API LLM, la traduction wolof->français n'est PAS exécutée
    # (elle nécessite NLLB). On vérifie seulement le flux FR.
    hybrid, filtered = _retrievers()
    r = answer_legacy("Quels sont les documents nécessaires ?", hybrid, filtered)
    assert "response_fr" in r
    assert r["response"] == r["response_fr"]  # entrée FR -> réponse FR


def test_answer_question_vide():
    hybrid, filtered = _retrievers()
    r = answer_legacy("   ", hybrid, filtered)
    assert r["response"]
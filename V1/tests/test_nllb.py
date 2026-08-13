"""Tests de la traduction NLLB réelle (marqués, chargent le modèle ~600M)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from translation.nllb import wolof_to_french, french_to_wolof, load_model

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NLLB_TESTS") != "1",
    reason="NLLB charges a ~600M model. Set RUN_NLLB_TESTS=1 to enable.",
)


@pytest.fixture(scope="module")
def model_loaded():
    load_model()
    return True


def test_wolof_to_french(model_loaded):
    result, seconds = wolof_to_french("dama beug wout kayitu juddu?")
    assert result
    assert "naissance" in result.lower() or "certificat" in result.lower()


def test_french_to_wolof(model_loaded):
    result, seconds = french_to_wolof("Comment obtenir un extrait de naissance ?")
    assert result
    assert "juddu" in result.lower()


def test_translation_roundtrip(model_loaded):
    fr, _ = wolof_to_french("Jërejëf lool ci dimbal bi.")
    assert fr.strip()
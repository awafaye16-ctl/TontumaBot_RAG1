"""Tests du TTS : le modèle wolof est chargé si disponible, sinon fallback edge-tts."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def test_synthesize_edge_fallback(tmp_path):
    """Sans modèle wolof configuré, le fallback edge-tts produit un fichier."""
    from config import settings

    old = settings.WOLOF_TTS_MODEL
    settings.WOLOF_TTS_MODEL = ""
    try:
        from tts.tts import synthesize, source
        out = tmp_path / "t.mp3"
        synthesize("Bonjour, test.", str(out))
        assert out.exists()
        assert out.stat().st_size > 1000
        assert source() == "edge-tts-fr"
    finally:
        settings.WOLOF_TTS_MODEL = old


def test_source_never_empty():
    """Le moteur TTS a toujours une valeur."""
    from tts.tts import source
    assert source() in ("speecht5-wolof", "edge-tts-fr")
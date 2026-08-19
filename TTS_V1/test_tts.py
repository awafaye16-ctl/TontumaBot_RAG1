"""Tests du pipeline TTS."""

import os
import numpy as np
import pytest

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "audio.wav")


@pytest.fixture(scope="module")
def tts():
    from tts import TTS
    return TTS(audio_path=AUDIO_PATH)


# ── Tests unitaires ────────────────────────────────────────────────

class TestRemoveArtifacts:
    def test_silent_input_stays_silent(self):
        from tts import remove_artifacts
        audio = np.zeros(8000)
        cleaned = remove_artifacts(audio)
        assert np.max(np.abs(cleaned)) < 0.01

    def test_output_not_clipped(self):
        from tts import remove_artifacts
        audio = np.random.randn(8000).astype(np.float32)
        cleaned = remove_artifacts(audio)
        assert np.max(np.abs(cleaned)) <= 1.0

    def test_does_not_modify_original(self):
        from tts import remove_artifacts
        audio = np.random.randn(1000).astype(np.float32)
        original = audio.copy()
        remove_artifacts(audio)
        np.testing.assert_array_equal(audio, original)

    def test_removes_high_freq(self):
        """Un signal large bande doit perdre ses hautes fréquences."""
        from tts import remove_artifacts
        from scipy.fft import rfft, rfftfreq
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        # Signal large bande : voix (300Hz) + grincement (6000Hz)
        audio = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)
        cleaned = remove_artifacts(audio)
        freqs = rfftfreq(len(cleaned), 1.0/16000)
        spec = np.abs(rfft(cleaned))
        lf = np.sqrt(np.mean(spec[(freqs > 200) & (freqs < 500)] ** 2))
        hf = np.sqrt(np.mean(spec[(freqs > 5500) & (freqs < 7000)] ** 2))
        # Le HF doit être plus réduit que le LF
        if lf > 0:
            assert hf / lf < 1.0

    def test_preserves_voice_band(self):
        """Un signal à 300Hz doit être préservé."""
        from tts import remove_artifacts
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t)
        cleaned = remove_artifacts(audio)
        assert np.max(np.abs(cleaned)) > 0.1


# ── Tests d'intégration ────────────────────────────────────────────

class TestModelLoading:
    def test_instantiation(self, tts):
        assert tts.model is not None
        assert tts.processor is not None
        assert tts.vocoder is not None
        assert tts.speaker_encoder is not None

    def test_device_detected(self, tts):
        assert tts.device.type in ("cpu", "cuda")

    def test_speaker_embedding_shape(self, tts):
        assert tts.speaker_embedding.shape == (1, 512)


class TestSpeakerEmbedding:
    def test_load_from_audio(self, tts):
        assert tts.speaker_embedding is not None


class TestSpeechGeneration:
    def test_returns_numpy(self, tts):
        audio = tts.generate("Bonjour")
        assert isinstance(audio, np.ndarray)

    def test_wolof(self, tts):
        audio = tts.generate("ñu ne ñoom ñooy nattukaay satélite yi")
        assert len(audio) > 0

    def test_french(self, tts):
        audio = tts.generate("Bonjour, bienvenue.")
        assert len(audio) > 0

    def test_not_silent(self, tts):
        audio = tts.generate("Test de volume")
        assert np.max(np.abs(audio)) > 0.01

    def test_not_clipped(self, tts):
        audio = tts.generate("Test de clipping")
        assert np.max(np.abs(audio)) <= 1.0


class TestFileOutput:
    def test_save_and_load(self, tts, tmp_path):
        import soundfile as sf
        out = tmp_path / "output.wav"
        tts.generate("Test", output_path=str(out))
        assert out.exists()
        data, sr = sf.read(str(out))
        assert sr == 16000
        assert len(data) > 0

"""TTS Wolof/Français avec SpeechT5 - Pipeline simplifié."""

import os
import torch
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt, medfilt, resample
import pywt
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from speechbrain.inference import EncoderClassifier

try:
    from IPython.display import Audio, display
except ImportError:
    Audio = display = None

SAMPLE_RATE = 16000


def _wavelet_denoise(audio, wavelet='db8', level=5):
    """Débruitage ondelettes : supprime les artefacts du vocoder en préservant la voix."""
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio

    coeffs = pywt.wavedec(audio, wavelet, level=level)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    if sigma < 1e-10:
        return audio

    for i in range(1, len(coeffs)):
        level_strength = i / len(coeffs)
        threshold = sigma * np.sqrt(2 * np.log(len(audio))) * (0.3 + 0.7 * level_strength)
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    rec = pywt.waverec(coeffs, wavelet).astype(np.float32)
    return rec[:len(audio)]


def _lowpass(audio, cutoff_hz, order, sr=SAMPLE_RATE):
    """Passe-bas Butterworth."""
    sos = butter(order, cutoff_hz / (sr / 2), btype="low", output="sos")
    return sosfilt(sos, audio)


def remove_artifacts(audio_np):
    """Pipeline complet pour éliminer le grincement du vocoder HiFi-GAN."""
    audio = audio_np.copy().astype(np.float32)

    # 1. Passe-haut 100Hz
    sos_hp = butter(4, 100 / (SAMPLE_RATE / 2), btype="high", output="sos")
    audio = sosfilt(sos_hp, audio)

    # 2. Débruitage ondelettes (atténue les hautes fréquences du grincement)
    audio = _wavelet_denoise(audio)

    # 3. Median filter kernel=5 (supprime les micro-coupures ponctuelles)
    audio = medfilt(audio, kernel_size=5).astype(np.float32)

    # 4. Normalisation RMS
    rms_orig = np.sqrt(np.mean(audio_np ** 2))
    rms_filt = np.sqrt(np.mean(audio ** 2))
    if rms_filt > 0.001 and rms_orig > 0.001:
        audio *= rms_orig / rms_filt

    # 5. Passe-bas 4200Hz ordre 10 (coupe le résidu HF)
    audio = _lowpass(audio, 4200, 10)

    # 6. Clip de sécurité
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio *= 0.95 / peak

    return audio


class TTS:
    """Pipeline TTS avec suppression de grincements et voix fluide."""

    def __init__(
        self,
        tts_checkpoint="bilalfaye/speecht5_tts-wolof-v0.2",
        vocoder_checkpoint="microsoft/speecht5_hifigan",
        speaker_encoder_source="speechbrain/spkrec-xvect-voxceleb",
        audio_path=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.processor = SpeechT5Processor.from_pretrained(tts_checkpoint)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(tts_checkpoint).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_checkpoint).to(self.device)

        self.speaker_encoder = EncoderClassifier.from_hparams(
            source=speaker_encoder_source,
            savedir="pretrained_models/speaker_encoder",
        )

        self.speaker_embedding = torch.randn(1, 512).to(self.device)

        if audio_path and os.path.exists(audio_path):
            self.load_speaker_embedding(audio_path)

    def load_speaker_embedding(self, audio_path):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = resample(audio, num_samples).astype(np.float32)
        signal = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            self.speaker_embedding = self.speaker_encoder.encode_batch(signal.to(self.device))
        self.speaker_embedding = self.speaker_embedding.squeeze(1)
        print(f"Speaker embedding chargé depuis {audio_path} ({self.speaker_embedding.shape})")

    def generate(self, text, output_path=None, num_beams=8, temperature=0.7,
                 no_repeat_ngram_size=3, repetition_penalty=1.3, max_length=200):
        inputs = self.processor(text=text, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        speech = self.model.generate(
            inputs["input_ids"],
            speaker_embeddings=self.speaker_embedding.to(self.device),
            vocoder=self.vocoder,
            num_beams=num_beams, temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )

        audio = speech.detach().cpu().numpy()
        audio = remove_artifacts(audio)

        if output_path:
            sf.write(output_path, audio, SAMPLE_RATE)
            print(f"Audio sauvegardé: {output_path}")
        return audio

    def play(self, text, **kwargs):
        audio = self.generate(text, **kwargs)
        if display and Audio:
            display(Audio(audio, rate=SAMPLE_RATE))
        return audio

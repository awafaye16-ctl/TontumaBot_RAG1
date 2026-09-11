import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tts_Ooleil import tts


class TTSFallbackTests(unittest.TestCase):
    def test_missing_speaker_reference_never_uses_random_voice(self):
        with patch.multiple(tts, _s5_ready=False, _s5_model=None, _s5_processor=None, _s5_vocoder=None, _s5_spk_emb=None), \
                patch('transformers.SpeechT5Processor.from_pretrained'), \
                patch('transformers.SpeechT5ForTextToSpeech.from_pretrained'), \
                patch('transformers.SpeechT5HifiGan.from_pretrained'), \
                patch('huggingface_hub.hf_hub_download', side_effect=RuntimeError('offline')), \
                patch.object(tts.os.path, 'exists', return_value=False), \
                patch.object(tts.torch, 'randn') as random:
            self.assertFalse(tts._load_speecht5())
            self.assertFalse(tts._s5_ready)
        random.assert_not_called()

    def test_incomplete_oolel_checkpoint_still_attempts_speecht5(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(tts.os.environ, {'OOLEL_MODEL_DIR': directory}), \
                patch.object(tts, 'require_model', side_effect=FileNotFoundError('Incomplete model')), \
                patch.object(tts, '_synth_oolel', return_value=False), \
                patch.object(tts, '_synth_speecht5', return_value=True) as fallback, \
                patch.object(tts.sf, 'info', return_value=SimpleNamespace(format='WAV', frames=16000)):
            result = tts.synthesize_with_metadata('Naka nga def?', str(Path(directory) / 'test.wav'), engine='oolel', language='wo')
        fallback.assert_called_once()
        self.assertEqual(result['engine'], 'speecht5-wolof')
        self.assertTrue(result['fallback'])
        self.assertIn('oolel', result['errors'])

    def test_wolof_never_falls_back_to_french_edge_voice(self):
        with patch.object(tts, '_synth_oolel', return_value=False), \
                patch.object(tts, '_synth_speecht5', return_value=False), \
                patch.object(tts, '_synth_edge') as edge, \
                patch.object(tts, 'require_model', return_value='test/model'):
            with self.assertRaises(RuntimeError):
                tts.synthesize_with_metadata('Naka nga def?', 'unused.wav', engine='oolel', language='wo')
        edge.assert_not_called()

    def test_french_fallback_reports_actual_engine(self):
        with patch.object(tts, '_synth_oolel', return_value=False), \
                patch.object(tts, '_synth_speecht5', return_value=False), \
                patch.object(tts, '_synth_edge'), \
                patch.object(tts, 'require_model', return_value='test/model'), \
                patch.object(tts.sf, 'info', return_value=SimpleNamespace(format='WAV', frames=24000)):
            result = tts.synthesize_with_metadata('Bonjour', 'unused.wav', engine='oolel', language='fr')
        self.assertEqual(result['engine'], 'edge-tts-fr')
        self.assertEqual(result['language'], 'fr')
        self.assertTrue(result['fallback'])


if __name__ == '__main__':
    unittest.main()

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config


class ModelAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def put(self, name, content=b"payload"):
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def checkpoint(self):
        self.put("config.json", b"{}")
        self.put("model.safetensors")

    def test_empty_invalid_and_lfs_config_rejected(self):
        self.checkpoint()
        for content in (b"", b"not json", b"[]", b"version https://git-lfs.github.com/spec/v1"):
            self.put("config.json", content)
            with self.subTest(content=content):
                self.assertIn("config.json", config.missing_model_files(self.path))

    def test_directory_instead_of_config_rejected(self):
        (self.path / "config.json").mkdir()
        self.put("model.safetensors")
        self.assertIn("config.json", config.missing_model_files(self.path))

    def test_oolel_empty_init_allowed_but_empty_code_and_weights_rejected(self):
        for name in ("config.json", "tokenizer.json"):
            self.put(name, b"{}")
        for name in ("modeling_oolel_voices.py", "configuration_oolel_voices.py",
                     "ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors"):
            self.put(name)
        self.put("models/__init__.py", b"")
        self.assertEqual(config.missing_model_files(self.path, "oolel"), [])
        for name in ("modeling_oolel_voices.py", "s3gen.safetensors"):
            self.put(name, b"")
            self.assertIn(name, config.missing_model_files(self.path, "oolel"))

    def test_all_shards_required_and_traversal_rejected(self):
        self.put("config.json", b"{}")
        for shards in (("a.safetensors", "b.safetensors"), ("../outside.bin",), ("C:\\outside.bin",)):
            self.put("model.safetensors.index.json", json.dumps({"weight_map": dict(enumerate(shards))}).encode())
            self.put("a.safetensors")
            self.assertTrue(config.missing_model_files(self.path))
        self.put("model.safetensors.index.json", b'{"weight_map":{"x":"a.safetensors"}}')
        self.assertEqual(config.missing_model_files(self.path), [])

    def test_local_status_does_not_infer_or_download(self):
        self.checkpoint()
        with patch("huggingface_hub.try_to_load_from_cache") as cache:
            status = config.model_status(str(self.path))
        cache.assert_not_called()
        self.assertTrue(status["files_available"])
        self.assertEqual(status["type"], "local")
        self.assertFalse(status["inference_tested"])

    def test_cache_uses_main_revision_not_arbitrary_snapshot(self):
        self.checkpoint()
        with patch("huggingface_hub.try_to_load_from_cache", return_value=str(self.path / "config.json")) as cache:
            status = config.model_status("test/model")
        cache.assert_called_once_with("test/model", "config.json", revision="main")
        self.assertEqual(status["type"], "cache_hf")
        self.assertTrue(status["files_available"])

    def test_missing_cache_is_not_claimed_available_online(self):
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            status = config.model_status("test/model")
        self.assertEqual(status["state"], "not_cached")
        self.assertFalse(status["files_available"])
        self.assertIsNone(status["remote_available"])

    def test_partial_cache_is_incomplete(self):
        self.put("config.json", b"{}")
        with patch("huggingface_hub.try_to_load_from_cache", return_value=str(self.path / "config.json")):
            status = config.model_status("test/model")
        self.assertEqual(status["state"], "incomplete")
        self.assertFalse(status["files_available"])

    def test_explicit_local_selection_is_not_masked_by_default_candidate(self):
        self.checkpoint()
        explicit = self.path / "chosen"
        with patch.dict(os.environ, {"TEST_LEGACY_PATH": str(explicit)}):
            selected = config._resolve_model((), ("TEST_LEGACY_PATH",), (self.path,), "test/model")
        self.assertEqual(selected, str(explicit))

    def test_default_hub_id_preserves_local_first(self):
        self.checkpoint()
        with patch.dict(os.environ, {"TEST_LEGACY_PATH": "test/model"}):
            selected = config._resolve_model((), ("TEST_LEGACY_PATH",), (self.path,), "test/model")
        self.assertEqual(selected, str(self.path))

    def test_custom_hub_id_is_not_replaced_by_unrelated_local_weights(self):
        self.checkpoint()
        with patch.dict(os.environ, {"TEST_LEGACY_PATH": "other/checkpoint"}):
            selected = config._resolve_model((), ("TEST_LEGACY_PATH",), (self.path,), "test/model")
        self.assertEqual(selected, "other/checkpoint")

    def test_auxiliary_models_require_processor_assets(self):
        self.checkpoint()
        for kind, required in (("speecht5", "spm_char.model"), ("embeddings", "modules.json"),
                               ("reranker", "tokenizer.json")):
            with self.subTest(kind=kind):
                self.assertIn(required, config.missing_model_files(self.path, kind))

    def test_cache_permission_error_is_not_reported_as_success(self):
        with patch("huggingface_hub.try_to_load_from_cache", side_effect=PermissionError):
            status = config.model_status("test/model")
        self.assertEqual(status["state"], "check_failed")
        self.assertFalse(status["files_available"])

    def test_downloader_check_uses_runtime_inventory(self):
        spec = importlib.util.spec_from_file_location("downloader_test", ROOT / "download_models.py")
        downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader)
        with patch.object(downloader, "configured_models", return_value={"stt_wo": ("test/model", "whisper")}), \
                patch.object(downloader, "model_status", return_value={"files_available": True, "state": "files_present", "type": "cache_hf", "path": "cache", "missing": []}), \
                patch.object(downloader, "snapshot_download") as download:
            downloader.main(["--check"])
        download.assert_not_called()


class STTInputTests(unittest.TestCase):
    def test_language_routing_is_explicit(self):
        from input import stt
        self.assertEqual(stt.source('fr'), config.settings.STT_FR_MODEL_DIR)
        self.assertEqual(stt.source('wo'), config.settings.STT_WO_MODEL_DIR)
        with self.assertRaises(ValueError):
            stt.resolve_language('unknown')

    def test_silent_and_long_audio_fail_before_loading_whisper(self):
        from input import stt
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder, patch.object(stt, 'load_model') as load:
            path = Path(folder) / 'fixture.wav'
            for samples in (np.zeros(16000), np.ones(31 * 16000) * 0.1):
                sf.write(path, samples, 16000)
                with self.assertRaises(ValueError):
                    stt.transcribe(str(path), 'fr')
        load.assert_not_called()

    def test_too_short_audio_rejected_before_loading_whisper(self):
        # Un enregistrement quasi vide (< 0.3 s) est rejeté explicitement plutôt que
        # transmis au modèle, où une entrée dégénérée peut provoquer un crash natif.
        from input import stt
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder, patch.object(stt, 'load_model') as load:
            path = Path(folder) / 'fixture.wav'
            sf.write(path, np.ones(int(0.1 * 16000)) * 0.5, 16000)
            with self.assertRaises(ValueError):
                stt.transcribe(str(path), 'fr')
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

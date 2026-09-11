import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import app as api


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()

    def test_routes_exist(self):
        paths = {route.path for route in api.app.routes}
        self.assertTrue({"/ask", "/ask/audio", "/ask/image", "/ws/ask", "/ws/stt", "/translate", "/session/{session_id}"} <= paths)
        for path in ("/", "/admin", "/docs", "/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_invalid_fields_rejected_before_models(self):
        with patch.object(api, "pipeline_answer") as answer:
            for payload in ({"question": " "}, {"question": "Bonjour", "language": "xx"},
                            {"question": "Bonjour", "provider": "unknown"},
                            {"question": "Bonjour", "with_audio": True}):
                self.assertEqual(self.client.post("/ask", json=payload).status_code, 422)
        answer.assert_not_called()

    def test_rest_forwards_forced_language(self):
        response = {"response": "test", "response_fr": "test", "trace": {"input_lang": "wo"}}
        with patch.object(api, "pipeline_answer", return_value=response) as answer:
            result = self.client.post("/ask", json={"question": "Question", "language": "wolof"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(answer.call_args.kwargs["input_lang"], "wo")

    def test_websocket_runs_to_done_with_language(self):
        async def fake_stream(question, **kwargs):
            self.assertEqual(kwargs["input_lang"], "wo")
            yield {"type": "lang", "lang": "wo"}
            yield {"type": "token", "text": "Réponse"}
            yield {"type": "done", "response_fr": "Réponse", "response": "Tontu", "trace": {}}

        with patch.object(api, "answer_stream", fake_stream):
            with self.client.websocket_connect("/ws/ask") as ws:
                ws.send_json({"question": "Question", "language": "wo"})
                messages = [ws.receive_json() for _ in range(3)]
        self.assertEqual(messages[-1]["type"], "done")
        self.assertEqual(messages[-1]["response"], "Tontu")

    def test_stream_error_is_not_successful_done(self):
        async def failing(*args, **kwargs):
            yield {"type": "lang", "lang": "fr"}
            raise RuntimeError("private provider detail")

        with patch.object(api, "answer_stream", failing):
            with self.client.websocket_connect("/ws/ask") as ws:
                ws.send_json({"question": "Question"})
                self.assertEqual(ws.receive_json()["type"], "lang")
                message = ws.receive_json()
        self.assertEqual(message["type"], "error")
        self.assertNotIn("private", message["text"])

    def test_audio_uses_worker_and_removes_only_own_upload(self):
        response = {"response": "test", "response_fr": "test", "trace": {"input_lang": "fr"}}
        with tempfile.TemporaryDirectory() as folder, patch.object(api, "UPLOAD_DIR", Path(folder)), patch.object(api, "_transcribe", return_value=("Question", {"text": "Question", "model": "test"})), patch.object(api, "pipeline_answer", return_value=response):
            result = self.client.post("/ask/audio", files={"file": ("micro.webm", b"fake audio", "audio/webm")}, data={"language": "fr"})
            self.assertEqual(list(Path(folder).iterdir()), [])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["trace"]["stt"]["text"], "Question")
        self.assertIn("total_latency_ms", result.json()["trace"])

    def test_invalid_audio_does_not_call_stt(self):
        with patch.object(api, "_transcribe") as stt, tempfile.TemporaryDirectory() as folder, patch.object(api, "UPLOAD_DIR", Path(folder)):
            self.assertEqual(self.client.post("/ask/audio", files={"file": ("empty.webm", b"", "audio/webm")}).status_code, 400)
            self.assertEqual(self.client.post("/ask/audio", files={"file": ("bad.exe", b"bad")}).status_code, 400)
        stt.assert_not_called()

    def test_stt_websocket_until_stop(self):
        with patch.object(api, "_transcribe_chunks", return_value=("Question", {"model": "test"})) as transcribe:
            with self.client.websocket_connect("/ws/stt?language=fr") as ws:
                ws.send_bytes(b"audio")
                ws.send_text("stop")
                message = ws.receive_json()
        self.assertEqual(message["type"], "final")
        self.assertEqual(message["text"], "Question")
        self.assertEqual(transcribe.call_args.args[1], "fr")

    def test_unique_audio_paths_and_public_urls(self):
        req = api.AskRequest(question="Question", tts=True)
        self.assertNotEqual(api._options(req)["tts_out"], api._options(req)["tts_out"])
        with tempfile.TemporaryDirectory() as folder, patch.object(api, "STATIC_DIR", Path(folder)):
            path = Path(folder) / "response_test.wav"
            path.write_bytes(b"RIFF")
            result = api._public_result({"audio": str(path)})
            self.assertEqual(result["audio"], "/static/response_test.wav")

    def test_admin_rejects_remote_mutations(self):
        with patch.dict(os.environ, {"ADMIN_API_KEY": ""}), patch.object(api.vectorstore, "clear_all") as clear:
            remote = TestClient(api.app, client=("203.0.113.1", 1234))
            self.assertEqual(remote.post("/admin/documents/clear").status_code, 403)
            remote.close()
        clear.assert_not_called()

    def test_missing_ocr_returns_actionable_error(self):
        with patch.object(api.shutil, 'which', return_value=None):
            response = self.client.post('/ask/image', files={'file': ('test.png', b'fixture', 'image/png')})
        self.assertEqual(response.status_code, 503)
        self.assertIn('Tesseract', response.json()['detail'])

    def test_models_uses_shared_file_audit_without_claiming_inference(self):
        status = {"type": "cache_hf", "files_available": True, "inference_tested": False}
        with patch.object(api, "model_status", return_value=status) as audit:
            result = self.client.get("/models")
        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(payload["status"], "files_checked_not_inference")
        for name in ("stt_wo", "stt_fr", "tts_speecht5", "tts_vocoder", "embeddings"):
            self.assertEqual(payload["models"][name]["status"], status)
        self.assertTrue(audit.called)
        self.assertFalse(payload["models"]["llm"]["inference_tested"])

    def test_health_does_not_claim_inference_tested(self):
        with patch.object(api.vectorstore, "count", return_value=0), patch.object(api.vectorstore, "all_documents", return_value=[]):
            result = self.client.get("/health")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["llm_status"], "configured_not_tested")
        self.assertEqual(result.json()["session"]["max_questions"], api.memoire.max_questions)
        self.assertEqual(result.json()["session"]["inactivity_timeout_seconds"], api.memoire.duree_vie_s)


if __name__ == "__main__":
    unittest.main()

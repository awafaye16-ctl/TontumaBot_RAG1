import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.getenv("RUN_LIVE_MODELS") == "1", "Opt-in real inference: RUN_LIVE_MODELS=1")
class LiveModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="tontuma-live-")

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def run_case(self, code, timeout=900):
        prelude = "import sys, os\nfrom pathlib import Path\nsys.path.insert(0, 'src')\nwork = Path(os.environ['LIVE_WORK'])\n"
        env = {**os.environ, "LIVE_WORK": self.workspace.name, "PYTHONUTF8": "1",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        result = subprocess.run([sys.executable, "-u", "-X", "utf8", "-c", prelude + textwrap.dedent(code)],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
        self.assertEqual(result.returncode, 0, "Real inference subprocess failed")

    def test_01_translation_routes(self):
        self.run_case('''
            from fastapi.testclient import TestClient
            import app
            client = TestClient(app.app)
            for direction, text in [('wo2fr', 'Ana mairie bi ?'), ('fr2wo', 'La mairie est dans le bâtiment A.')]:
                response = client.get('/translate', params={'direction': direction, 'text': text})
                assert response.status_code == 200, response.json()
                data = response.json()
                assert data['output'].strip()
                print(direction, data['model'], '=>', data['output'])
            client.close()
        ''')

    def test_02_embeddings_reranker(self):
        self.run_case('''
            import numpy as np
            from vectorstore import get_embedder
            from retrieval.reranker import rerank
            vectors = get_embedder().encode(['Où est la mairie ?', 'La mairie est dans le bâtiment A.'])
            assert vectors.shape == (2, 384) and np.isfinite(vectors).all()
            candidates = ['La mairie est dans le bâtiment A.', 'La température du soleil est très élevée.']
            results = rerank('Où est la mairie ?', candidates, top_k=2)
            assert results[0][0] == 0, results
            print('Embeddings:', vectors.shape, 'Reranker:', results)
        ''')

    def test_03_edge_french(self):
        self.run_case('''
            from tts_Ooleil.tts import synthesize_with_metadata
            import soundfile as sf
            result = synthesize_with_metadata('Où est la mairie ?', str(work / 'fr.wav'), engine='edge', language='fr')
            assert result['engine'] == 'edge-tts-fr' and not result['fallback'], result
            info = sf.info(result['path'])
            assert 0 < info.duration < 30 and info.format == 'WAV'
            print('Edge FR:', result['engine'], info.duration, 'seconds')
        ''')

    def test_04_oolel_wolof(self):
        self.run_case('''
            from tts_Ooleil.tts import synthesize_with_metadata
            import soundfile as sf
            result = synthesize_with_metadata('Naka nga def?', str(work / 'wo.wav'), engine='oolel', language='wo')
            assert result['engine'] == 'oolel-voices' and not result['fallback'], result
            info = sf.info(result['path'])
            assert 0 < info.duration < 30 and info.format == 'WAV'
            print('Oolel WO:', result['engine'], info.duration, 'seconds')
        ''')

    def test_05_speecht5_wolof(self):
        self.run_case('''
            from tts_Ooleil.tts import synthesize_with_metadata
            import soundfile as sf
            result = synthesize_with_metadata('Naka nga def?', str(work / 'speecht5.wav'), engine='speecht5', language='wo')
            assert result['engine'] == 'speecht5-wolof' and not result['fallback'], result
            info = sf.info(result['path'])
            assert info.duration > 0 and info.format == 'WAV'
            print('SpeechT5 WO:', result['engine'], info.duration, 'seconds')
        ''')

    def test_06_stt_both_languages(self):
        self.run_case('''
            from input.stt import transcribe, source
            for language in ('fr', 'wo'):
                path = work / (language + '.wav')
                assert path.is_file(), 'TTS fixture missing: ' + language
                text = transcribe(str(path), language=language)
                assert text.strip()
                if language == 'fr':
                    assert 'mairie' in text.lower(), text
                print('STT', language, source(language), '=>', text)
        ''')

    def test_07_rag_api_and_websockets(self):
        self.run_case('''
            import uuid
            import chromadb
            from fastapi.testclient import TestClient
            import app
            from config import settings
            import vectorstore
            vectorstore._client = chromadb.EphemeralClient()
            vectorstore.COLLECTION = 'live_' + uuid.uuid4().hex
            vectorstore._collection = None
            app.STATIC_DIR = work
            app.UPLOAD_DIR = work / 'uploads'
            settings.SEED_ENABLED = False
            app.app.dependency_overrides[app.require_admin] = lambda: None
            client = TestClient(app.app)
            empty = client.post('/ask', json={'question': 'Où est la mairie ?', 'language': 'fr'})
            assert empty.status_code == 200, empty.json()
            assert not empty.json()['trace']['llm']['called']
            assert not empty.json()['sources']
            added = client.post('/admin/documents', data={'text': 'Document fictif de test. La mairie est située dans le bâtiment A.', 'title': 'Live fixture'})
            assert added.status_code == 200, added.json()
            listing = client.get('/admin/documents').json()
            assert listing['total_chunks'] > 0
            rest = client.post('/ask', json={'question': 'Où est la mairie ?', 'language': 'fr'})
            assert rest.status_code == 200, rest.json()
            result = rest.json()
            assert result['sources'] and result['trace']['retrieval']['n_candidates']
            print('REST FR:', result['response'])
            from generation.llm import ABSTENTION
            absent = client.post('/ask', json={'question': 'Combien coûte un passeport ?', 'language': 'fr'})
            assert absent.status_code == 200, absent.json()
            assert absent.json()['trace']['retrieval']['n_candidates'] > 0
            assert absent.json()['response_fr'] == ABSTENTION, absent.json()['response_fr']
            print('Unrelated question: retrieval attempted, grounded abstention OK')
            with client.websocket_connect('/ws/ask') as ws:
                ws.send_json({'question': 'Où est la mairie ?', 'language': 'fr'})
                tokens = []
                while True:
                    event = ws.receive_json()
                    assert event['type'] != 'error', event
                    if event['type'] == 'token':
                        tokens.append(event['text'])
                    if event['type'] == 'done':
                        assert tokens and event['sources']
                        assert ''.join(tokens).strip() == event['response_fr']
                        print('WebSocket FR done:', event['response'])
                        break
            if not (work / 'fr.wav').is_file():
                from tts_Ooleil.tts import synthesize_with_metadata
                synthesize_with_metadata('Où est la mairie ?', str(work / 'fr.wav'), engine='edge', language='fr')
            with (work / 'fr.wav').open('rb') as audio:
                response = client.post('/ask/audio', files={'file': ('fr.wav', audio, 'audio/wav')}, data={'language': 'fr'})
            assert response.status_code == 200, response.json()
            assert 'mairie' in response.json()['trace']['stt_text'].lower()
            print('REST audio FR:', response.json()['response'])
            with client.websocket_connect('/ws/stt?language=fr') as ws:
                ws.send_bytes((work / 'fr.wav').read_bytes())
                ws.send_text('stop')
                event = ws.receive_json()
                assert event['type'] == 'final' and 'mairie' in event['text'].lower(), event
                print('WebSocket STT:', event['text'])
            wolof = client.post('/ask', json={'question': 'Ana mairie bi ?', 'language': 'wo', 'tts': True})
            assert wolof.status_code == 200, wolof.json()
            result = wolof.json()
            assert result['response_wo'] and result['trace']['wolof_to_french']['result']
            assert result['trace']['tts']['status'] == 'ok', result.get('warnings')
            assert result['audio'].startswith('/static/response_')
            audio = client.get('/response.wav', params={'filename': Path(result['audio']).name})
            assert audio.status_code == 200 and audio.content[:4] == b'RIFF'
            print('REST WO+TTS:', result['response_wo'], result['trace']['tts']['engine'])
            for doc in listing['documents']:
                deleted = client.delete('/admin/documents/' + doc['id'])
                assert deleted.status_code == 200, deleted.json()
            assert client.get('/admin/documents').json()['total_chunks'] == 0
            print('Admin add/list/delete: temporary collection only, OK')
            client.close()
        ''', timeout=1200)


if __name__ == '__main__':
    unittest.main()

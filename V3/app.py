import asyncio
import hmac
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from config import settings, configured_models, model_status
from pipeline import answer as pipeline_answer, answer_stream, normalize_language
from memory.conversation import memoire, SessionExpired, SessionLimitReached
from data.seed_docs import get_documents
import vectorstore
import ingestion

logger = logging.getLogger("tontuma")
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
Provider = Literal["groq", "gemini", "local"]
TTSEngine = Literal["oolel", "speecht5", "edge"]
SEED_DOCS, SEED_FILTERED = get_documents()


def _auto_index_documents():
    if vectorstore.count() > 0:
        return
    dataset = settings.HF_DOCUMENTS_DATASET
    if dataset:
        ingestion.ingest_huggingface_dataset(dataset, split=settings.HF_DATASET_SPLIT,
            text_column=settings.HF_TEXT_COLUMN, title_column=settings.HF_TITLE_COLUMN,
            source_column=settings.HF_SOURCE_COLUMN)
        return
    docs_dir = BASE_DIR / "data" / "documents"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.iterdir()):
            if path.suffix.lower() in (".pdf", ".md", ".txt"):
                ingestion.ingest_file(str(path), title=path.stem)


@asynccontextmanager
async def lifespan(application):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    application.state.database_ready = False
    try:
        await asyncio.to_thread(_auto_index_documents)
        application.state.database_ready = True
    except Exception as exc:
        logger.error("Initialisation documentaire échouée (%s)", type(exc).__name__)
    if getattr(settings, "PRELOAD_TTS", False):
        from tts_Ooleil.tts import _load_oolel
        try:
            loaded = await asyncio.to_thread(_load_oolel)
            logger.info("Préchargement Oolel : %s", "prêt" if loaded else "indisponible")
        except Exception as exc:
            logger.warning("Préchargement Oolel échoué (%s)", type(exc).__name__)
    if getattr(settings, "PRELOAD_MODELS", False):
        await _preload_models()
    yield


async def _preload_models():
    """Charge STT (FR+WO), embeddings, reranker et NLLB au démarrage.

    Chaque modèle est chargé indépendamment : l'échec de l'un n'empêche pas les
    autres de se précharger, ni le serveur de démarrer (dégradation silencieuse,
    identique au comportement au premier appel réel).
    """
    from input.stt import load_model as load_stt
    from vectorstore import get_embedder
    from retrieval.reranker import load_model as load_reranker
    from translation.nllb import _load_wo_fr, _load_fr_wo

    steps = [
        ("STT français", lambda: load_stt("fr")),
        ("STT wolof", lambda: load_stt("wo")),
        ("Embeddings", get_embedder),
        ("Reranker", load_reranker),
        ("NLLB wolof→français", _load_wo_fr),
        ("NLLB français→wolof", _load_fr_wo),
    ]
    for label, loader in steps:
        try:
            await asyncio.to_thread(loader)
            logger.info("Préchargement %s : prêt", label)
        except Exception as exc:
            logger.warning("Préchargement %s échoué (%s)", label, type(exc).__name__)


app = FastAPI(title="TontumaBot V3", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=12000)
    provider: Provider | None = None
    tts: bool = False
    tts_engine: TTSEngine | None = None
    language: str | None = None
    session_id: str | None = None

    @field_validator("language")
    @classmethod
    def validate_language(cls, value):
        return normalize_language(value)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question vide")
        return value.strip()


class RagasRequest(BaseModel):
    test_cases: list[dict] = Field(min_length=1, max_length=50)
    provider: Provider = "groq"


def require_admin(request: Request):
    token = os.getenv("ADMIN_API_KEY", "")
    if token:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + token):
            raise HTTPException(401, "Authentification administrateur requise")
    elif request.client is None or request.client.host not in ("127.0.0.1", "::1", "testclient"):
        raise HTTPException(403, "Administration distante désactivée : configurer ADMIN_API_KEY")


def _service_error(exc):
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, SessionLimitReached):
        return HTTPException(409, str(exc))
    if isinstance(exc, SessionExpired):
        return HTTPException(410, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, "Entrée invalide : vérifier la langue, le texte et le format du fichier")
    status = getattr(exc, "status_code", None)
    if status == 429:
        return HTTPException(429, "Quota du fournisseur atteint. Réessayez ultérieurement.")
    logger.warning("Traitement indisponible (%s)", type(exc).__name__)
    return HTTPException(503, "Traitement indisponible : vérifier la configuration et les fichiers des modèles")


def _options(req, audio_path=None):
    use_seed = getattr(settings, "SEED_ENABLED", False)
    return {"provider": req.provider or settings.LLM_PROVIDER, "tts": req.tts,
            "tts_engine": req.tts_engine, "input_lang": req.language,
            "session_id": req.session_id,
            "tts_out": str(audio_path or STATIC_DIR / f"response_{uuid.uuid4().hex}.wav"),
            "seed_docs": SEED_DOCS if use_seed else None,
            "seed_filtered": SEED_FILTERED if use_seed else None}


def _public_result(result):
    if result.get("audio"):
        path = Path(result["audio"]).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            raise RuntimeError("Sortie audio absente ou invalide")
        result["audio"] = "/static/" + path.name
    return result


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/health")
def health():
    try:
        count = vectorstore.count()
        documents = vectorstore.all_documents()
    except Exception as exc:
        raise _service_error(exc) from exc
    ready = getattr(app.state, "database_ready", True)
    if not ready:
        raise HTTPException(503, "Initialisation documentaire incomplète")
    return {"status": "ok", "version": "3.0.0", "llm_provider": settings.LLM_PROVIDER,
            "llm_ready": settings.llm_ready,
            "llm_status": "configured_not_tested" if settings.llm_ready else "not_configured",
            "session": {"max_questions": memoire.max_questions,
                 "inactivity_timeout_seconds": memoire.duree_vie_s},
            "tts_engine": settings.TTS_ENGINE, "nllb_wo_fr": settings.NLLB_WO_FR_MODEL,
            "nllb_fr_wo": settings.NLLB_FR_WO_MODEL, "stt_model": settings.STT_MODEL_PATH,
            "reranker": settings.RERANKER_MODEL, "n_documents": len(documents), "n_chunks": count}


@app.get("/models")
def models():
    result = {name: {"source": source, "status": model_status(source, kind)}
              for name, (source, kind) in configured_models().items()}
    result["stt"] = result["stt_wo"]
    result["llm"] = {"provider": settings.LLM_PROVIDER, "model": settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else settings.LOCAL_LLM_MODEL if settings.LLM_PROVIDER == "local" else "gemini-2.5-flash", "configured": settings.llm_ready, "inference_tested": False}
    result["tts"] = {"engine": settings.TTS_ENGINE, "oolel_repo": settings.OOLEL_TTS_REPO, "speecht5_fb": settings.WOLOF_TTS_MODEL}
    result["session"] = {"max_questions": memoire.max_questions,
                          "inactivity_timeout_seconds": memoire.duree_vie_s}
    result["llm"]["availability"] = "not_tested" if settings.llm_ready else "not_configured"
    result["tts"]["edge_availability"] = "remote_not_tested"
    return {"status": "files_checked_not_inference", "models": result,
            "features": {"ocr": {"executable_found": shutil.which("tesseract") is not None,
                                  "engine": "tesseract", "inference_tested": False},
                         "stt": {"max_duration_seconds": settings.STT_MAX_DURATION_SECONDS,
                                 "automatic_language_detection": False}}}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        if req.session_id is None:
            req.session_id = str(uuid.uuid4())
        result = _public_result(pipeline_answer(req.question, **_options(req)))
        result["session_id"] = req.session_id
        return result
    except Exception as exc:
        raise _service_error(exc) from exc


async def _save_upload(file, suffixes):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in suffixes:
        raise HTTPException(400, "Format de fichier non supporté")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / (uuid.uuid4().hex + suffix)
    total = 0
    try:
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Fichier trop volumineux (max 50 Mo)")
                output.write(chunk)
        if not total:
            raise HTTPException(400, "Fichier vide")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return path


STT_WORKER_TIMEOUT_SECONDS = 180
# Isoler chaque appel dans un sous-processus recharge le modèle à froid à
# chaque fois (~90-110s au lieu de ~25s à chaud) : un coût élevé à payer sur
# toutes les requêtes. En pratique, les crashs natifs observés se sont
# systématiquement produits pendant la transcription WOLOF (jamais français) ;
# on isole donc uniquement le wolof, qui reste rapide et fiable en français.
STT_ISOLATED_LANGUAGES = {"wo"}


def _transcribe(path, language):
    """Transcrit ; isole la langue à risque (wolof) dans un sous-processus.

    L'inférence Whisper peut occasionnellement provoquer un crash natif
    (segfault) selon l'audio fourni ; ce crash ne doit jamais emporter le
    serveur applicatif entier. Isolé dans un sous-processus, il ne se traduit
    plus que par un échec propre de cette seule requête.
    """
    from input.stt import source, resolve_language
    started = time.perf_counter()
    selected = resolve_language(language)
    if selected not in STT_ISOLATED_LANGUAGES:
        from input.stt import transcribe
        text = transcribe(str(path), language=selected)
        if not text.strip():
            raise ValueError("Aucun texte transcrit")
        return text, {"text": text, "model": source(selected), "language_requested": language,
                      "language_selected": selected, "automatic_language_detection": False,
                      "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    worker = str(BASE_DIR / "src" / "input" / "stt_worker.py")
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", worker, str(path), selected],
            capture_output=True, text=True, encoding="utf-8", timeout=STT_WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"La transcription a dépassé {STT_WORKER_TIMEOUT_SECONDS}s ; réessayez.") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Le processus de transcription s'est arrêté de manière inattendue (code {proc.returncode}). "
            "Réessayez, éventuellement avec un autre enregistrement."
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except ValueError as exc:
        raise RuntimeError("Réponse de transcription invalide.") from exc
    if "error" in payload:
        raise ValueError(payload["error"])
    text = payload.get("text", "")
    if not text.strip():
        raise ValueError("Aucun texte transcrit")
    return text, {"text": text, "model": source(selected), "language_requested": language,
                  "language_selected": selected, "automatic_language_detection": False,
                  "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def _media_request(tts, tts_engine, provider, language, session_id=None):
    try:
        return AskRequest(question="media", tts=tts, tts_engine=tts_engine, provider=provider,
                          language=language, session_id=session_id)
    except ValidationError:
        raise HTTPException(422, "Paramètres de langue, fournisseur ou TTS invalides")


@app.post("/ask/audio")
async def ask_audio(file: UploadFile = File(...), tts: bool = Form(False), tts_engine: TTSEngine | None = Form(None),
                    provider: Provider | None = Form(None), language: str | None = Form(None),
                    session_id: str | None = Form(None)):
    if session_id is None:
        session_id = str(uuid.uuid4())
    req = _media_request(tts, tts_engine, provider, language, session_id=session_id)
    path = await _save_upload(file, {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".aac", ".flac"})
    started = time.perf_counter()
    try:
        text, stt_trace = await asyncio.to_thread(_transcribe, path, req.language)
        result = await asyncio.to_thread(pipeline_answer, text, **_options(req))
        result["trace"].update({"stt": stt_trace, "stt_text": text, "input_type": "audio",
                                "stt_language_forced": req.language is not None,
                                "audio_language": result["trace"]["input_lang"],
                                "total_latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        result = _public_result(result)
        result["session_id"] = session_id
        return result
    except Exception as exc:
        raise _service_error(exc) from exc
    finally:
        path.unlink(missing_ok=True)


def _ocr(path):
    from PIL import Image
    executable = shutil.which("tesseract")
    if not executable:
        raise HTTPException(503, "OCR indisponible : installer Tesseract et ses langues fra/eng, puis vérifier le PATH du serveur.")
    with Image.open(path) as image:
        if image.width * image.height > 25000000:
            raise ValueError("Image trop grande")
        image.verify()
    result = subprocess.run([executable, str(path), "stdout", "-l", os.getenv("OCR_LANGS", "fra+eng")],
                            capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    if not result.stdout.strip():
        raise ValueError("Aucun texte extrait")
    return result.stdout.strip()


@app.post("/ask/image")
async def ask_image(file: UploadFile = File(...), tts: bool = Form(False), tts_engine: TTSEngine | None = Form(None),
                    provider: Provider | None = Form(None), language: str | None = Form(None),
                    session_id: str | None = Form(None)):
    if session_id is None:
        session_id = str(uuid.uuid4())
    req = _media_request(tts, tts_engine, provider, language, session_id=session_id)
    path = await _save_upload(file, {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})
    started = time.perf_counter()
    try:
        text = await asyncio.to_thread(_ocr, path)
        result = await asyncio.to_thread(pipeline_answer, text, **_options(req))
        result["trace"].update({"input_type": "image", "ocr_text": text,
                                "total_latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        result["session_id"] = session_id
        return _public_result(result)
    except Exception as exc:
        raise _service_error(exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.get("/translate")
def translate_test(text: str = "Jërejëf lool ci dimbal bi.", direction: Literal["wo2fr", "fr2wo"] = "wo2fr"):
    from translation.nllb import wolof_to_french, french_to_wolof
    if not text.strip() or len(text) > 12000:
        raise HTTPException(400, "Texte vide ou trop long")
    try:
        fn = french_to_wolof if direction == "fr2wo" else wolof_to_french
        result, seconds = fn(text)
    except Exception as exc:
        raise _service_error(exc) from exc
    return {"input": text, "output": result, "direction": "fr→wo" if direction == "fr2wo" else "wo→fr",
            "model": settings.NLLB_FR_WO_MODEL if direction == "fr2wo" else settings.NLLB_WO_FR_MODEL, "seconds": seconds}


@app.delete("/session/{session_id}")
async def effacer_session(session_id: str):
    memoire.effacer_session(session_id)
    return {"ok": True}


@app.get("/session/{session_id}")
def statut_session(session_id: str):
    return memoire.statut_session(session_id)


@app.get("/response.wav")
def get_audio(filename: str | None = None):
    if not filename:
        raise HTTPException(400, "Utilisez l'URL audio propre à la réponse, ou indiquez filename")
    if not filename.startswith("response_") or Path(filename).name != filename or not filename.endswith(".wav"):
        raise HTTPException(400, "Nom audio invalide")
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "Audio introuvable")
    return FileResponse(path, media_type="audio/wav")


@app.post("/admin/documents", dependencies=[Depends(require_admin)])
async def add_document(file: UploadFile | None = File(None), text: str | None = Form(None), title: str | None = Form(None)):
    if file is None and (text is None or not text.strip()):
        raise HTTPException(400, "Fournissez un fichier ou du texte")
    if file is not None:
        original_name = Path(file.filename or "document").name
        path = await _save_upload(file, {".txt", ".md", ".pdf"})
        try:
            content = await asyncio.to_thread(ingestion.extract_text, str(path))
            n = await asyncio.to_thread(ingestion.ingest_text, content, title or original_name, original_name)
        except Exception as exc:
            raise _service_error(exc) from exc
        finally:
            path.unlink(missing_ok=True)
    else:
        if len(text) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Texte trop volumineux")
        n = await asyncio.to_thread(ingestion.ingest_text, text, title or "Document manuel")
    if not n:
        raise HTTPException(400, "Aucun texte exploitable")
    return {"ok": True, "chunks": n, "title": title or (original_name if file is not None else "Document manuel")}


@app.get("/admin/documents", dependencies=[Depends(require_admin)])
def list_documents():
    return {"documents": vectorstore.all_documents(), "total_chunks": vectorstore.count()}


@app.delete("/admin/documents/{document_id}", dependencies=[Depends(require_admin)])
def remove_document(document_id: str):
    n = vectorstore.delete_document(document_id)
    if n == 0:
        raise HTTPException(404, "Document introuvable")
    return {"ok": True, "deleted_chunks": n}


@app.post("/admin/documents/clear", dependencies=[Depends(require_admin)])
def clear_documents():
    return {"ok": True, "deleted_chunks": vectorstore.clear_all()}


@app.post("/eval/ragas", dependencies=[Depends(require_admin)])
def eval_ragas(req: RagasRequest):
    from evaluation.ragas_eval import evaluate_pipeline, TestCase
    from dataclasses import asdict
    try:
        cases = [TestCase(**tc) for tc in req.test_cases]
        for case in cases:
            case.language = normalize_language(case.language) or "fr"
        summary = evaluate_pipeline(cases, provider=req.provider)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "Cas d'évaluation invalides") from exc
    except Exception as exc:
        raise _service_error(exc) from exc
    return {"summary": asdict(summary), "method": "local_lexical_heuristics", "output_file": None}


@app.websocket("/ws/ask")
async def ws_ask(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=60)
        if len(raw) > 20000:
            raise ValueError("Requête trop longue")
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("type") == "question":
            data.pop("type")
        req = AskRequest.model_validate(data)
        if req.session_id is None:
            req.session_id = str(uuid.uuid4())
        async for event in answer_stream(req.question, **_options(req)):
            if event["type"] == "done":
                event = _public_result(event)
                event["session_id"] = req.session_id
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        error = _service_error(exc)
        await websocket.send_json({"type": "error", "text": error.detail, "status": error.status_code})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.websocket("/ws/stt")
async def ws_stt(websocket: WebSocket):
    await websocket.accept()
    chunks = []
    total = 0
    language = None
    try:
        language = normalize_language(websocket.query_params.get("language"))
        while True:
            msg = await asyncio.wait_for(websocket.receive(), timeout=60)
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("text") == "stop":
                break
            if msg.get("bytes"):
                total += len(msg["bytes"])
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError("Enregistrement trop volumineux")
                chunks.append(msg["bytes"])
            elif msg.get("text"):
                config = json.loads(msg["text"])
                language = normalize_language(config.get("language"))
        if not chunks:
            raise ValueError("Audio vide")
        text, trace = await _transcribe_chunks(chunks, language)
        await websocket.send_json({"type": "final", "text": text, "trace": {"stt": trace}})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        error = _service_error(exc)
        await websocket.send_json({"type": "error", "text": error.detail, "status": error.status_code})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


async def _transcribe_chunks(chunks, language=None):
    def run():
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as output:
            output.write(b"".join(chunks))
            path = Path(output.name)
        try:
            return _transcribe(path, language)
        finally:
            path.unlink(missing_ok=True)
    return await asyncio.to_thread(run)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

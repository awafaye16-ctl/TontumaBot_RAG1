"""TontumaBot V2 — FastAPI avec RAG + Reranker + TTS dual engine."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "data"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from pipeline import answer as pipeline_answer
from seed_docs import get_documents
import vectorstore
import ingestion

app = FastAPI(title="TontumaBot V2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SEED_DOCS, SEED_FILTERED = get_documents()


class AskRequest(BaseModel):
    question: str
    provider: str | None = None
    tts: bool = False
    tts_engine: str | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_ready": settings.llm_ready,
        "tts_engine": settings.TTS_ENGINE,
        "reranker": settings.RERANKER_MODEL,
        "n_documents": len(vectorstore.all_documents()),
        "n_chunks": vectorstore.count(),
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question vide")
    provider = req.provider or settings.LLM_PROVIDER
    return pipeline_answer(
        req.question.strip(),
        provider=provider,
        tts=req.tts,
        tts_engine=req.tts_engine,
        tts_out=os.path.join(STATIC_DIR, "response.wav"),
        seed_docs=SEED_DOCS,
        seed_filtered=SEED_FILTERED,
    )


@app.post("/ask/audio")
async def ask_audio(
    file: UploadFile = File(...),
    tts: bool = Form(False),
    tts_engine: str | None = Form(None),
):
    from input.stt import transcribe

    tmp = os.path.join(BASE_DIR, "uploads", file.filename)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(await file.read())
    try:
        text = transcribe(tmp, language=settings.STT_LANGUAGE or None)
    except Exception as e:
        raise HTTPException(500, f"STT échoué : {e}")
    return pipeline_answer(
        text,
        provider=settings.LLM_PROVIDER,
        tts=tts,
        tts_engine=tts_engine,
        tts_out=os.path.join(STATIC_DIR, "response.wav"),
        seed_docs=SEED_DOCS,
        seed_filtered=SEED_FILTERED,
    )


@app.get("/translate")
def translate_test(text: str = "Jërejëf lool ci dimbal bi.", direction: str = "wo2fr"):
    from translation.nllb import wolof_to_french, french_to_wolof
    if direction == "fr2wo":
        result, seconds = french_to_wolof(text)
        return {"input": text, "output": result, "direction": "fr->wo", "seconds": seconds}
    result, seconds = wolof_to_french(text)
    return {"input": text, "output": result, "direction": "wo->fr", "seconds": seconds}


@app.post("/admin/documents")
async def add_document(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    title: str | None = Form(None),
):
    if file is None and (text is None or not text.strip()):
        raise HTTPException(400, "Fournissez un fichier ou un texte")
    if file is not None:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {".txt", ".md", ".pdf"}:
            raise HTTPException(400, f"Format non supporté : {ext}")
        path = os.path.join(BASE_DIR, "uploads", file.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(await file.read())
        try:
            n = ingestion.ingest_file(path, title=title)
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        if n == 0:
            raise HTTPException(400, "Aucun texte exploitable")
        return {"ok": True, "chunks": n, "title": title or file.filename}
    n = ingestion.ingest_text(text, title or "Document manuel")
    if n == 0:
        raise HTTPException(400, "Texte vide")
    return {"ok": True, "chunks": n, "title": title or "Document manuel"}


@app.get("/admin/documents")
def list_documents():
    return {"documents": vectorstore.all_documents(), "total_chunks": vectorstore.count()}


@app.delete("/admin/documents/{document_id}")
def remove_document(document_id: str):
    n = vectorstore.delete_document(document_id)
    if n == 0:
        raise HTTPException(404, "Document introuvable")
    return {"ok": True, "deleted_chunks": n}


@app.post("/admin/documents/clear")
def clear_documents():
    n = vectorstore.clear_all()
    return {"ok": True, "deleted_chunks": n}


@app.get("/response.wav")
def get_audio():
    path = os.path.join(STATIC_DIR, "response.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "Aucun audio")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

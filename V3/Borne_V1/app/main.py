"""Point d'entrée FastAPI de TONTOUMA-BOT Borne_V1.

L'application expose :
  - /docs (Swagger UI) ;
  - /redoc (ReDoc) ;
  - /api/sessions/* (gestion des sessions).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.dependencies import close_redis
from app.routes import sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_redis()


app = FastAPI(
    title="TONTOUMA-BOT Borne_V1",
    description="Assistant hospitalier multimodal — gestion sécurisée des sessions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(sessions.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirige la racine vers la documentation Swagger."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}

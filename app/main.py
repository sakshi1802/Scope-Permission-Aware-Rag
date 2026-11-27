import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingest import load_chunks
from app.llm import GeminiEmbedder, generate
from app.rag import answer_question
from app.rbac import ROLES
from app.store import ChunkStore

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = os.getenv("DOCS_DIR", ROOT / "docs")
FRONTEND_DIST = ROOT / "frontend" / "dist"
TOP_K = int(os.getenv("TOP_K", 4))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.55))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", 500))


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ChunkStore(GeminiEmbedder())
    store.index(load_chunks(DOCS_DIR))
    app.state.store = store
    yield


app = FastAPI(title="Scope", lifespan=lifespan)


class AskRequest(BaseModel):
    role: str
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/roles")
def roles():
    return [{"id": role_id, "label": label} for role_id, label in ROLES.items()]


@app.get("/api/docs")
def docs(role: str):
    if role not in ROLES:
        raise HTTPException(400, "unknown role")
    return app.state.store.docs_for_role(role)


@app.post("/api/ask")
def ask(body: AskRequest):
    if body.role not in ROLES:
        raise HTTPException(400, "unknown role")
    try:
        return answer_question(
            body.question.strip(), body.role, app.state.store, generate,
            top_k=TOP_K, min_score=MIN_SCORE,
        )
    except Exception:
        raise HTTPException(503, "The model is unavailable right now. Please try again.")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
"""
main.py — Yap-Doc web server

Serves:
  • /api/upload   POST   — upload + embed a document for the current session
  • /api/chat     POST   — ask a question, get an answer back
  • /api/document DELETE — forget the currently loaded document
  • /api/models   GET    — list available model labels
  • /             GET    — the static frontend (static/index.html etc.)

Sessions are kept in a simple in-memory dict, keyed by a client-generated
X-Session-Id header. That's enough for a single-instance Render deploy.
If you scale to multiple instances or need documents to survive a restart,
swap SESSIONS for something like Redis.
"""

import os
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend import load_document, get_answer, GROQ_MODELS, GEMINI_MODELS

load_dotenv()

app = FastAPI(title="Yap-Doc API")

# Open CORS — harmless for a same-origin deploy, and means you can also
# point a separately-hosted frontend at this API later if you want to.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> {"vectorstore": ..., "filename": ..., "chunks": int}
SESSIONS: dict = {}


def _session_id(x_session_id: Optional[str]) -> str:
    return x_session_id or "default"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    model_choice: str = "Groq"
    deep_research: bool = False
    history: List[ChatMessage] = []


@app.get("/api/models")
def list_models():
    return {"groq": list(GROQ_MODELS.keys()), "gemini": list(GEMINI_MODELS.keys())}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = _session_id(x_session_id)

    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in ("pdf", "docx", "txt"):
        raise HTTPException(400, "Only PDF, DOCX, or TXT files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "That file looks empty.")

    try:
        vectorstore, num_chunks = load_document(file.filename, file_bytes)
    except Exception as e:
        raise HTTPException(500, f"Couldn't process that document: {e}")

    SESSIONS[session_id] = {
        "vectorstore": vectorstore,
        "filename": file.filename,
        "chunks": num_chunks,
    }
    return {"filename": file.filename, "chunks": num_chunks}


@app.delete("/api/document")
def clear_document(x_session_id: Optional[str] = Header(default=None)):
    SESSIONS.pop(_session_id(x_session_id), None)
    return {"status": "cleared"}


@app.post("/api/chat")
async def chat(req: ChatRequest, x_session_id: Optional[str] = Header(default=None)):
    session_id = _session_id(x_session_id)
    vectorstore = SESSIONS.get(session_id, {}).get("vectorstore")

    history = [{"role": m.role, "content": m.content} for m in req.history]

    answer = get_answer(
        query=req.query,
        model_choice=req.model_choice,
        vectorstore=vectorstore,
        deep_research=req.deep_research,
        chat_history=history,
    )
    return {"answer": answer}


# ── Serve the frontend (must be registered after the API routes above) ──────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

import os
import glob
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz import fuzz

from schemas import AgentResponse

BASE_DIR = Path(__file__).resolve().parent


def _resolve_dir(env_value: str, default_name: str) -> Path:
    """Абсолютный путь: env → BASE_DIR/default. Работает из любой cwd."""
    if env_value:
        p = Path(env_value)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p.resolve()
    return (BASE_DIR / default_name).resolve()


KB_DIR = _resolve_dir(os.getenv("KB_DIR", ""), "knowledge_base")

app = FastAPI(title="ПСБ Юридический Ассистент")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = None


def get_agent():
    """Ленивая инициализация: сервер поднимется, даже если index ещё не собран."""
    global _agent
    if _agent is None:
        from agent import LegalAssistantAgent
        try:
            _agent = LegalAssistantAgent()
        except TypeError:
            from retriever import LegalRetriever
            _agent = LegalAssistantAgent(retriever=LegalRetriever())
    return _agent


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]


def _call_agent(messages: List[Dict[str, str]]):
    agent = get_agent()
    if hasattr(agent, "answer_from_messages"):
        return agent.answer_from_messages(messages)
    if hasattr(agent, "process_query"):
        return agent.process_query(messages[-1]["content"] if messages else "")
    raise RuntimeError("Agent has neither answer_from_messages nor process_query")


def _read_doc_text(doc_id: str):
    agent = None
    try:
        agent = get_agent()
    except Exception:
        agent = None
    if agent is not None and hasattr(agent, "get_raw_document_text"):
        try:
            return agent.get_raw_document_text(doc_id)
        except Exception:
            pass
    fpath = KB_DIR / doc_id
    if not fpath.exists():
        return None
    return fpath.read_text(encoding="utf-8", errors="replace")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>UI not ready</h1>", status_code=200)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/documents")
def list_documents():
    files = sorted(KB_DIR.glob("*.txt")) if KB_DIR.exists() else []
    docs = []
    for fpath in files:
        fname = fpath.name
        category = "Документы компании" if fname.startswith("документы") else "Законодательство РФ"
        clean_title = (
            fname.replace("документы компании__", "")
            .replace("законы__", "")
            .replace(".txt", "")
        )
        docs.append({"doc_id": fname, "title": clean_title, "category": category})
    return {"documents": docs}


@app.post("/api/chat", response_model=AgentResponse)
def chat_endpoint(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Пустая история сообщений")
    try:
        return _call_agent(req.messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.get("/api/document/{doc_id}")
def get_document(doc_id: str, highlight_quote: str = ""):
    safe_name = Path(doc_id).name
    doc_text = _read_doc_text(safe_name)
    if doc_text is None:
        raise HTTPException(status_code=404, detail="Документ не найден")

    paragraphs = doc_text.split("\n\n")
    best_idx, best_score = -1, 0
    if highlight_quote.strip():
        for i, p in enumerate(paragraphs):
            if len(p.strip()) < 20:
                continue
            score = fuzz.partial_ratio(highlight_quote.lower(), p.lower())
            if score > best_score:
                best_score, best_idx = score, i

    return {
        "doc_id": safe_name,
        "paragraphs": paragraphs,
        "highlight_index": best_idx if best_score >= 65 else -1,
        "match_score": best_score,
    }

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
from rapidfuzz import fuzz

from agent import LegalAssistantAgent
from schemas import AgentResponse

app = FastAPI(title="ПСБ Юридический Ассистент")

# Инициализируем единого агента один раз при старте сервера
agent = LegalAssistantAgent()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]

@app.get("/", response_class=HTMLResponse)
def index():
    """Отдаем одностраничный интерфейс без всякого React"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat", response_model=AgentResponse)
def chat_endpoint(req: ChatRequest):
    """Общение фронтенда с агентом"""
    # Передаем историю диалога в того же агента, который работает в solution.py
    return agent.answer_from_messages(req.messages)

@app.get("/api/document/{doc_id}")
def get_document(doc_id: str, highlight_quote: str = ""):
    """
    Возвращает текст документа из базы знаний и находит точные координаты
    подтверждающего фрагмента для подсветки (обязательное требование допуска!)
    """
    doc_text = agent.get_raw_document_text(doc_id)
    if doc_text is None:
        raise HTTPException(status_code=404, detail="Документ не найден")

    # Ищем лучший совпадающий абзац через rapidfuzz (из разрешенного requirements.txt)
    paragraphs = doc_text.split("\n\n")
    best_idx = -1
    best_score = 0

    if highlight_quote.strip():
        for i, p in enumerate(paragraphs):
            score = fuzz.partial_ratio(highlight_quote.lower(), p.lower())
            if score > best_score and len(p.strip()) > 20:
                best_score = score
                best_idx = i

    return {
        "doc_id": doc_id,
        "paragraphs": paragraphs,
        "highlight_index": best_idx if best_score >= 65 else -1,
        "match_score": best_score
    }
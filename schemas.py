from pydantic import BaseModel
from typing import List, Optional

class Citation(BaseModel):
    doc_id: str          # Имя файла документа в базе знаний
    doc_title: str       # Название документа / закона
    clause: str          # Пункт / статья (например, "п. 3.2" или "ст. 421 ГК РФ")
    quote: str           # Точная цитата из чанка для подсветки в UI
    score: float         # Скор реранкера Qwen3 Reranker 8B

class AgentResponse(BaseModel):
    answer: str                          # Итоговый текст ответа (идет и в CSV, и в чат)
    status: str                          # CONFIRMED | PARTIAL | REFUSAL_OLD_LAW | PREMISE_ERROR | OUT_OF_DOMAIN
    confidence: float                    # Итоговая уверенность [0..1]
    missing_info: Optional[str] = None   # Чего именно не хватает в базе или вопросе
    action_steps: List[str] = []         # Пошаговый план действий (бонус по п. 06 кейса)
    citations: List[Citation] = []       # Подтверждающие положения
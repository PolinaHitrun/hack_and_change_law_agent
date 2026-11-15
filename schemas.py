from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    clause: str = ""
    quote: str
    score: float = 0.0


class AgentResponse(BaseModel):
    answer: str
    status: str = "CONFIRMED"
    confidence: float = 0.0
    missing_info: Optional[str] = None
    action_steps: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

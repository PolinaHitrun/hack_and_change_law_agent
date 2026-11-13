import os
import json
import re
from openai import OpenAI
from schemas import AgentResponse, Citation
from prompts import ROUTER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT

# Импортируем класс Полины (он называется LegalRetriever в её коде)
try:
    from retriever import LegalRetriever
except ImportError:
    LegalRetriever = None

class LegalAssistantAgent:
    def __init__(self, retriever=None):
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
        self.model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-v4.1-flash"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
        # Инициализируем ретривер Полины
        self.retriever = retriever or (LegalRetriever() if LegalRetriever else None)

    def _parse_json(self, text: str) -> dict:
        """Очистка ответа нейросети от markdown и парсинг JSON"""
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        return {}

    def analyze_query(self, raw_text: str) -> dict:
        """Роутер (Шаг 1)"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text}
                ]
            )
            return self._parse_json(resp.choices[0].message.content)
        except Exception as e:
            print(f"Router Error: {e}")
            return {"standalone_query": raw_text, "is_safe": True, "target_date": None, "claims_to_check": []}

    def answer_from_messages(self, messages: list) -> AgentResponse:
        """Главный метод для работы с фронтендом (вызывается из app.py)"""
        
        # Склеиваем историю диалога
        raw_question = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # 1. Роутер
        route = self.analyze_query(raw_question)
        if not route.get("is_safe", True):
            return AgentResponse(
                answer="Я являюсь корпоративным юридическим ассистентом и могу отвечать только на правовые вопросы в рамках предоставленной базы знаний.",
                status="OUT_OF_DOMAIN",
                confidence=1.0
            )

        standalone_query = route.get("standalone_query", raw_question)
        target_date = route.get("target_date")
        claims = route.get("claims_to_check", [])

        # 2. Поиск документов (вызов метода Полины)
        chunks = []
        if self.retriever:
            try:
                # Полина возвращает просто список чанков (не кортеж!)
                chunks = self.retriever.retrieve(query=standalone_query, target_date=target_date)
            except Exception as e:
                print(f"Retriever Error: {e}")

        # 3. Упаковка в теги
        evidence_blocks = []
        for i, ch in enumerate(chunks):
            evidence_blocks.append(
                f"<document_evidence id='{i+1}' file='{ch.get('doc_id', '')}' clause='{ch.get('clause_id', '')}' edition='{ch.get('status', '')}'>\n"
                f"{ch.get('contextual_text', '')}\n"
                f"</document_evidence>"
            )
        evidence_str = "\n\n".join(evidence_blocks)

        user_prompt = (
            f"Исходный запрос / диалог:\n{raw_question}\n\n"
            f"Уточненный вопрос: {standalone_query}\n"
            f"Целевая дата проверки редакции: {target_date or 'Действующая на сегодня'}\n"
            f"Утверждения пользователя для проверки: {claims}\n\n"
            f"Документы из базы знаний:\n{evidence_str if evidence_str else 'ДОКУМЕНТЫ НЕ НАЙДЕНЫ'}"
        )

        # 4. Генерация ответа
        try:
            gen_resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            res_dict = self._parse_json(gen_resp.choices[0].message.content)
        except Exception as e:
            print(f"Generator Error: {e}")
            res_dict = {"answer": "Произошла ошибка при генерации ответа.", "status": "OUT_OF_DOMAIN", "confidence": 0.0}

        # 5. Сборка Citations
        final_citations = []
        for cit in res_dict.get("citations", []):
            final_citations.append(
                Citation(
                    doc_id=cit.get("doc_id", ""),
                    doc_title=cit.get("doc_title", "Документ"),
                    clause=cit.get("clause", ""),
                    quote=cit.get("quote", ""),
                    score=float(cit.get("score", 0.9))
                )
            )

        # 6. Возврат ответа
        return AgentResponse(
            answer=res_dict.get("answer", "Извините, не удалось сформулировать ответ."),
            status=res_dict.get("status", "CONFIRMED"),
            confidence=float(res_dict.get("confidence", 0.9)),
            missing_info=res_dict.get("missing_info"),
            action_steps=res_dict.get("action_steps", []),
            citations=final_citations
        )

    def get_raw_document_text(self, doc_id: str) -> str:
        """
        Восстановление полного текста документа из чанков Полины.
        Это нужно для того, чтобы на сайте работала желтая подсветка (вызов из app.py).
        """
        if not self.retriever or not hasattr(self.retriever, "chunks"):
            return "Текст документа недоступен (retriever не инициализирован)."
        
        # Собираем все чанки, относящиеся к этому документу
        doc_chunks = [c for c in self.retriever.chunks if c["doc_id"] == doc_id]
        if not doc_chunks:
            return None
            
        # Сортируем по порядку и склеиваем
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        full_text = "\n\n".join([c["raw_text"] for c in doc_chunks])
        return full_text
# ПСБ Юридический ассистент

RAG-ассистент по базе нормативных документов ПСБ.

## Архитектура
- `app.py` — FastAPI-сервер и веб-интерфейс
- `retriever.py` — гибридный поиск по базе знаний
- `agent.py` — LLM-агент, формирующий ответ и проверяющий достоверность
- `schemas.py` — общие Pydantic-модели
- `solution.py` — скрипт автосабмита для платформы
- `evaluate_local.py` — локальная проверка на demo_questions.csv

## Запуск одной командой
```bash
cp .env.example .env
docker compose up --build

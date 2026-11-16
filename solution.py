import os
import sys
import time
import zipfile
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FALLBACK_ANSWER = (
    "На основе предоставленной базы знаний невозможно сформировать "
    "достоверный ответ по данному запросу без дополнительных сведений."
)

BASE_DIR = Path(__file__).resolve().parent


def resolve_dir(env_value: str, default_name: str) -> Path:
    if env_value:
        p = Path(env_value)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p.resolve()
    return (BASE_DIR / default_name).resolve()


def ensure_knowledge_base(data_dir: Path, extract_to: Path) -> Path:
    """Распаковывает knowledge-base.zip в KB_DIR. Возвращает итоговую папку."""
    if extract_to.exists() and any(extract_to.iterdir()):
        return extract_to
    zip_path = data_dir / "knowledge-base.zip"
    if not zip_path.exists():
        # не падаем: возможно, индекс уже вшит в сабмит
        print(f"[WARN] {zip_path} не найден — работаем с тем, что есть в {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        return extract_to

    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # защита от zip-slip и от «внешней папки» в архиве
        for member in zf.namelist():
            target = (extract_to / member).resolve()
            if not str(target).startswith(str(extract_to.resolve())):
                raise RuntimeError(f"Недопустимый путь в архиве: {member}")
        zf.extractall(extract_to)

    # если архив пришёл с одной внешней папкой — поднимаем содержимое наверх
    entries = list(extract_to.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not any(extract_to.glob("*.txt")):
        inner = entries[0]
        for child in inner.iterdir():
            child.rename(extract_to / child.name)
        inner.rmdir()

    return extract_to


def _call_agent(agent, question_text: str):
    if hasattr(agent, "process_query"):
        return agent.process_query(str(question_text))
    if hasattr(agent, "answer_from_messages"):
        return agent.answer_from_messages([{"role": "user", "content": str(question_text)}])
    raise RuntimeError("Agent has neither process_query nor answer_from_messages")


def process_single_row(agent, row_id, question_text: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            res = _call_agent(agent, question_text)
            ans = (getattr(res, "answer", "") or "").strip()
            if ans:
                print(f"[INFO] id={row_id} ({attempt+1}/{max_retries}): got answer ({len(ans)} chars)")
                return {"id": row_id, "answer": ans[:9900]}
        except Exception as exc:
            wait_sec = 2 * (attempt + 1)
            print(f"[WARN] id={row_id} ({attempt+1}/{max_retries}): {exc}. Ждём {wait_sec}s")
            time.sleep(wait_sec)
    return {"id": row_id, "answer": FALLBACK_ANSWER}


def main() -> int:
    import pandas as pd  # импорт здесь, чтобы упасть только при реальном запуске

    data_dir = resolve_dir(os.getenv("DATA_DIR", ""), "data")
    kb_dir = resolve_dir(os.getenv("KB_DIR", ""), "knowledge_base")

    questions_path = data_dir / "questions.csv"
    if not questions_path.exists():
        for candidate in (BASE_DIR / "demo_questions.csv", Path("demo_questions.csv")):
            if candidate.exists():
                questions_path = candidate
                break
    if not questions_path.exists():
        print(f"[ERROR] questions.csv не найден ни в {data_dir}, ни рядом со скриптом")
        return 1

    # Сообщаем агенту/ретриверу, где база — до их импорта
    kb_dir = ensure_knowledge_base(data_dir, kb_dir)
    os.environ["KB_DIR"] = str(kb_dir)
    os.environ.setdefault("DATA_DIR", str(data_dir))

    print("1. Загружаем агента...")
    try:
        from retriever import LegalRetriever
    except Exception:
        LegalRetriever = None
    from agent import LegalAssistantAgent

    try:
        retriever = LegalRetriever() if LegalRetriever else None
        agent = LegalAssistantAgent(retriever=retriever) if retriever else LegalAssistantAgent()
    except TypeError:
        agent = LegalAssistantAgent()

    print(f"2. Читаем вопросы из {questions_path}...")
    df_questions = pd.read_csv(questions_path, encoding="utf-8-sig")
    if "id" not in df_questions.columns:
        print("[ERROR] в questions.csv нет колонки id")
        return 1

    q_col = next((c for c in ("question", "query", "text") if c in df_questions.columns), None)
    if q_col is None:
        print("[ERROR] в questions.csv нет колонки question/query/text")
        return 1

    max_workers = max(1, min(int(os.getenv("MAX_WORKERS", "2") or 2), 4))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_row, agent, row["id"], str(row[q_col])): row["id"]
            for _, row in df_questions.iterrows()
        }
        for future in as_completed(futures):
            results.append(future.result())

    df_results = pd.DataFrame(results)
    df_out = df_questions[["id"]].merge(df_results, on="id", how="left")
    df_out["answer"] = df_out["answer"].fillna(FALLBACK_ANSWER)
    df_out = df_out[["id", "answer"]]

    out_path = Path.cwd() / "submission.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"3. Готово! {len(df_out)} строк → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

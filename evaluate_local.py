import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def _call_agent(agent, question: str):
    if hasattr(agent, "process_query"):
        return agent.process_query(question)
    return agent.answer_from_messages([{"role": "user", "content": question}])


def run_eval(path: str = "demo_questions.csv"):
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

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

    rows = []
    for _, row in df.iterrows():
        question = str(row["question"])
        t0 = time.time()
        try:
            res = _call_agent(agent, question)
            answer = getattr(res, "answer", "")
            status = getattr(res, "status", "UNKNOWN")
            citations = getattr(res, "citations", []) or []
        except Exception as exc:
            answer, status, citations = f"ERROR: {exc}", "ERROR", []
        dt = round(time.time() - t0, 2)

        sources = ", ".join(f"{c.doc_title} ({c.clause})" for c in citations)
        print(f"\n=== ID {row['id']} ({dt}s) | {status} ===")
        print(f"ВОПРОС: {question}\nОТВЕТ: {answer}\nССЫЛКИ: {sources}")

        rows.append({"id": row["id"], "question": question, "status": status,
                     "sources": sources, "answer": answer, "time_sec": dt})

    out = Path.cwd() / "local_eval_report.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    run_eval()

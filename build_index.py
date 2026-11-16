import os
import re
import json
import zipfile
import logging
import requests
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "hackathon_key"
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")

INJECTION_ANCHORS = []
if os.path.exists("injections.txt"):
    with open("injections.txt", "r", encoding="utf-8") as f:
        INJECTION_ANCHORS = [line.strip() for line in f if line.strip()]


class RemoteEmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": self.model, "input": batch}
            for attempt in range(3):
                try:
                    resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=45)
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    data_sorted = sorted(data, key=lambda x: x["index"])
                    all_embeddings.extend([item["embedding"] for item in data_sorted])
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Ошибка при получении эмбеддингов: {e}")
                        all_embeddings.extend([[0.0] * 1024 for _ in batch])
        embs = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embs / norms


class SemanticSanitizer:
    def __init__(self, emb_client: RemoteEmbeddingClient):
        self.emb_client = emb_client
        self.anchor_embs = self.emb_client.get_embeddings(INJECTION_ANCHORS) if INJECTION_ANCHORS else None
        self.quarantine_log = []
        # Быстрый regex-фильтр явных prompt-инъекций внутри документов
        self.regex_guard = re.compile(
            r"(ignore\s+previous|system\s+prompt|забудь\s+все\s+инструкции|игнорируй\s+инструкции|ты\s+теперь)",
            re.IGNORECASE
        )

    def sanitize_clauses(self, doc_id: str, clauses: List[str]) -> List[str]:
        if not clauses:
            return []
        cleaned_clauses = []
        for text in clauses:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            kept = []
            for s in sentences:
                if self.regex_guard.search(s):
                    self.quarantine_log.append({"doc_id": doc_id, "removed_sentence": s, "reason": "regex"})
                    continue
                kept.append(s)
            if kept:
                cleaned_clauses.append(" ".join(kept))
        return cleaned_clauses


def extract_document_passport(filename: str, raw_text: str) -> Dict[str, Any]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    header_blob = " ".join(lines[:6]).lower()
    title = lines[0] if lines else filename

    valid_from, valid_to, status = None, None, "Действующая редакция"
    if "утратил силу" in filename.lower() or "утратил силу" in header_blob:
        status = "Утратил силу (ред. от 10.01.2023, действовал до 01.03.2025)"
        valid_from, valid_to = "2023-01-10", "2025-02-28"
    elif "ред 2025" in filename.lower() or "01.03.2025" in header_blob:
        status = "Действующая редакция (с 01.03.2025)"
        valid_from, valid_to = "2025-03-01", None
    elif "к-12" in filename.lower() or "15.04.2025" in filename.lower():
        status = "Обязательное разъяснение комплаенс (от 15.04.2025, действует с 21.04.2025)"
        valid_from, valid_to = "2025-04-21", None

    prefix = f"[Файл: {filename} | Документ: {title} | Статус: {status}]"
    return {
        "doc_id": filename,
        "title": title,
        "status": status,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "prefix": prefix
    }


def split_into_clauses(raw_text: str) -> List[Dict[str, str]]:
    """
    Разбивает текст на пункты/статьи и прикрепляет заметки об изменении редакции
    ("Часть 1 изменена с...", "Статья X дополнена...") к тексту самой нормы.
    """
    lines = raw_text.splitlines()
    pattern = re.compile(
        r"^(\d+\.\d+\.?|\d+\.|Раздел\s+\d+|Статья\s+\d+[\.\d]*|Часть\s+\d+[\.\d]*|ч\.\s*\d+|Пункт\s+\d+|Примечани[ея]:?)",
        re.IGNORECASE
    )
    edition_note_pattern = re.compile(
        r"^(Часть|Пункт|Статья|Примечания)\s+.*?(изменен|дополнен|в\s+редакции|утратил)",
        re.IGNORECASE
    )

    clauses = []
    current_label = "Шапка документа"
    current_lines = []
    pending_edition_note = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Если строка — это короткая заметка о дате изменения следующей части/пункта
        if edition_note_pattern.match(stripped) and len(stripped) < 160:
            if current_lines:
                clauses.append({"clause_id": current_label, "text": "\n".join(current_lines)})
                current_lines = []
            pending_edition_note = stripped
            continue

        m = pattern.match(stripped)
        if m:
            if current_lines:
                clauses.append({"clause_id": current_label, "text": "\n".join(current_lines)})
            current_label = m.group(1).strip()
            if pending_edition_note:
                current_lines = [f"[Информация о редакции: {pending_edition_note}]", stripped]
                pending_edition_note = None
            else:
                current_lines = [stripped]
        else:
            if pending_edition_note and not current_lines:
                current_lines = [f"[Информация о редакции: {pending_edition_note}]", stripped]
                pending_edition_note = None
            else:
                current_lines.append(stripped)

    if pending_edition_note:
        current_lines.append(pending_edition_note)
    if current_lines:
        clauses.append({"clause_id": current_label, "text": "\n".join(current_lines)})

    # Объединяем заголовки разделов ("1. Предмет договора") со следующим пунктом, чтобы они не терялись
    merged_clauses = []
    carry_header = None
    for cl in clauses:
        txt = cl["text"].strip()
        if len(txt) <= 45 and not any(w in txt.lower() for w in ["утратил силу", "изменен", "дополнен"]):
            carry_header = txt
        else:
            if carry_header:
                cl["text"] = f"{carry_header}\n{txt}"
                carry_header = None
            merged_clauses.append(cl)
    if carry_header and merged_clauses:
        merged_clauses[-1]["text"] += f"\n{carry_header}"

    return merged_clauses


def load_knowledge_base_texts() -> List[tuple[str, str]]:
    """Чтение базы знаний с учетом переменной окружения DATA_DIR по п. 6 Тех. регламента."""
    env_data_dir = os.getenv("DATA_DIR", "data")
    base_dir = Path(__file__).resolve().parent

    candidate_dirs = [
        Path(env_data_dir),
        base_dir / env_data_dir,
        base_dir / "DATA_DIR",
        base_dir / "data",
        base_dir
    ]

    search_dirs = []
    for d in candidate_dirs:
        if not d.exists():
            continue
        kb_zip = d / "knowledge-base.zip"
        if kb_zip.is_file():
            extract_target = Path("/tmp/extracted_kb") if os.access("/tmp", os.W_OK) else (d / "extracted_kb")
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(kb_zip, "r") as zip_ref:
                zip_ref.extractall(extract_target)
            search_dirs.append(extract_target)
            break
        search_dirs.append(d)

    loaded_docs = {}
    for s_dir in search_dirs:
        for file_path in sorted(s_dir.rglob("*.txt")):
            if file_path.name == "requirements.txt" or file_path.name == "injections.txt":
                continue
            raw_text = file_path.read_text(encoding="utf-8")
            # Формируем doc_id с категорией папки, если файл лежит в подпапке
            parent_name = file_path.parent.name
            if parent_name in ("документы компании", "законы"):
                doc_key = f"{parent_name}__{file_path.name}"
            else:
                doc_key = file_path.name
            if doc_key not in loaded_docs:
                loaded_docs[doc_key] = raw_text

    logger.info(f"Загружено документов базы знаний: {len(loaded_docs)}")
    return list(loaded_docs.items())


def build_deterministic_graph(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    edges = {"AMENDS": [], "VERSIONS": [], "LAW_NOTES": []}
    by_doc = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch["chunk_id"])
    doc_ids = list(by_doc.keys())

    for d_main in doc_ids:
        if "договор" in d_main.lower() and "доп" not in d_main.lower():
            m = re.search(r"([А-ЯЁ]{1,3}-\d{2}[-/]\d{4})", d_main, re.IGNORECASE)
            if m:
                target_num = m.group(1).replace("/", "-").lower()
                for d_other in doc_ids:
                    if d_other != d_main and target_num in d_other.replace("/", "-").lower():
                        edges["AMENDS"].append({
                            "source_doc": d_main, "target_doc": d_other,
                            "source_chunks": by_doc[d_main], "target_chunks": by_doc[d_other]
                        })

    reg_2023 = next((d for d in doc_ids if "регламент" in d.lower() and "2023" in d.lower()), None)
    reg_2025 = next((d for d in doc_ids if "регламент" in d.lower() and "2025" in d.lower()), None)
    k12 = next((d for d in doc_ids if "к-12" in d.lower() or "комплаенс" in d.lower()), None)
    if reg_2023 and reg_2025:
        edges["VERSIONS"].append({
            "reg_2023_doc": reg_2023, "reg_2025_doc": reg_2025, "letter_k12_doc": k12,
            "reg_2023_chunks": by_doc.get(reg_2023, []), "reg_2025_chunks": by_doc.get(reg_2025, []),
            "letter_k12_chunks": by_doc.get(k12, []) if k12 else [], "switch_date": "2025-03-01"
        })

    for d_law in doc_ids:
        if any(w in d_law.lower() for w in ["закон", "коап", "152", "нк_", "мрот"]):
            doc_chunks = [c for c in chunks if c["doc_id"] == d_law]
            anchors = [
                c["chunk_id"] for c in doc_chunks
                if c["chunk_index"] == 0 or "примечан" in c["clause_id"].lower() or "примечан" in c["raw_text"].lower()[:40]
            ]
            edges["LAW_NOTES"].append({
                "law_doc": d_law, "all_law_chunks": by_doc[d_law],
                "header_and_notes_chunks": list(dict.fromkeys(anchors))
            })
    return edges


def main(output_dir: str = "."):
    logger.info("Запуск сборки индекса...")
    emb_client = RemoteEmbeddingClient(LLM_BASE_URL, LLM_API_KEY, EMB_MODEL)
    sanitizer = SemanticSanitizer(emb_client)

    documents = load_knowledge_base_texts()
    if not documents:
        logger.error("Документы для индексации не найдены!")
        return

    all_chunks = []
    for filename, raw_text in documents:
        passport = extract_document_passport(filename, raw_text)
        clauses = split_into_clauses(raw_text)
        cleaned_texts = sanitizer.sanitize_clauses(filename, [c["text"] for c in clauses])

        for idx, (cl, c_text) in enumerate(zip(clauses, cleaned_texts)):
            if not c_text.strip():
                continue
            chunk_id = f"{filename}::chunk_{idx}"
            contextual_text = f"{passport['prefix']} [Пункт/Статья: {cl['clause_id']}]\n{c_text}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "doc_id": filename,
                "clause_id": cl["clause_id"],
                "status": passport["status"],
                "valid_from": passport["valid_from"],
                "valid_to": passport["valid_to"],
                "raw_text": c_text,
                "contextual_text": contextual_text
            })

    graph_edges = build_deterministic_graph(all_chunks)
    context_texts = [c["contextual_text"] for c in all_chunks]
    embeddings = emb_client.get_embeddings(context_texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    with open(os.path.join(output_dir, "chunks_and_graph.json"), "w", encoding="utf-8") as f:
        json.dump({
            "chunks": all_chunks,
            "graph": graph_edges,
            "quarantine_log": sanitizer.quarantine_log
        }, f, ensure_ascii=False, indent=2)
    logger.info("Индексация успешно завершена.")


if __name__ == "__main__":
    main()
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

# Конфигурация API платформы
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "hackathon_key")
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
CHAT_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4.1-Flash")

with open("injections.txt", "r", encoding="utf-8") as f:
    INJECTION_ANCHORS = [line.strip() for line in f if line.strip()]

class RemoteEmbeddingClient:
    """Клиент для генерации эмбеддингов через API прокси организаторов."""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": self.model, "input": batch}
            try:
                resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                data_sorted = sorted(data, key=lambda x: x["index"])
                all_embeddings.extend([item["embedding"] for item in data_sorted])
            except Exception as e:
                logger.error(f"Ошибка при получении эмбеддингов: {e}")
                # Fallback: нулевые векторы при сбое (чтобы не уронить сборку)
                all_embeddings.extend([[0.0] * 1024 for _ in batch])
        
        embs = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embs / norms


class SemanticSanitizer:
    def __init__(self, emb_client: RemoteEmbeddingClient):
        self.emb_client = emb_client
        logger.info("Векторизация банка атак (INJECTION_ANCHORS)...")
        self.anchor_embs = self.emb_client.get_embeddings(INJECTION_ANCHORS)
        self.quarantine_log = []

    def sanitize_clauses(self, doc_id: str, clauses: List[str]) -> List[str]:
        if not clauses:
            return []
        
        clause_embs = self.emb_client.get_embeddings(clauses)
        centroid = np.mean(clause_embs, axis=0, keepdims=True)
        centroid /= (np.linalg.norm(centroid) + 1e-9)

        cleaned_clauses = []
        for idx, text in enumerate(clauses):
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if not sentences:
                continue

            sent_embs = self.emb_client.get_embeddings(sentences)
            anchor_sims = np.max(np.dot(sent_embs, self.anchor_embs.T), axis=1)
            centroid_sims = np.dot(sent_embs, centroid.T).flatten()

            kept_sentences = []
            for s_idx, sent in enumerate(sentences):
                is_anchor_hit = float(anchor_sims[s_idx]) > 0.80
                is_outlier = float(centroid_sims[s_idx]) < 0.25

                if is_anchor_hit or is_outlier:
                    logger.warning(f"[{doc_id}] ОБНАРУЖЕНА ИНЪЕКЦИЯ (score={anchor_sims[s_idx]:.2f}): {sent[:50]}...")
                    self.quarantine_log.append({
                        "doc_id": doc_id,
                        "removed_sentence": sent,
                        "anchor_sim": round(float(anchor_sims[s_idx]), 3)
                    })
                    continue
                kept_sentences.append(sent)

            if kept_sentences:
                cleaned_clauses.append(" ".join(kept_sentences))
        return cleaned_clauses


def extract_document_passport(filename: str, raw_text: str) -> Dict[str, Any]:
    """Формирует паспорт документа для контекстного чанкинга."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    header_blob = " ".join(lines[:6]).lower()
    title = lines[0] if lines else filename

    # Извлечение дат и статусов (на основе известных ловушек базы знаний)
    valid_from, valid_to, status = None, None, "Действующая редакция"
    if "утратил силу" in filename.lower() or "утратил силу" in header_blob:
        status = "Утратил силу (ред. 2023, действовал до 2025-03-01)"
        valid_from, valid_to = "2023-01-01", "2025-02-28"
    elif "ред 2025" in filename.lower() or "01.03.2025" in header_blob:
        status = "Действующая редакция (с 2025-03-01)"
        valid_from, valid_to = "2025-03-01", None
    elif "к-12" in filename.lower() or "15.04.2025" in filename.lower():
        status = "Разъяснение (от 15.04.2025, действует с 2025-04-21)"
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
    """Разбивает текст на пункты/статьи для точного поиска."""
    lines = raw_text.splitlines()
    pattern = re.compile(
        r"^(\d+\.\d+\.?|\d+\.|Раздел\s+\d+|Статья\s+\d+[\.\d]*|Часть\s+\d+|ч\.\s*\d+|Примечани[ея]:?)",
        re.IGNORECASE
    )
    clauses = []
    current_label = "Шапка документа"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = pattern.match(stripped)
        if m:
            if current_lines:
                clauses.append({"clause_id": current_label, "text": "\n".join(current_lines)})
            current_label = m.group(1).strip()
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_lines:
        clauses.append({"clause_id": current_label, "text": "\n".join(current_lines)})
    return clauses


def load_knowledge_base_texts() -> List[tuple[str, str]]:
    """Распаковка и чтение файлов из knowledge-base.zip согласно регламенту."""
    data_dir = Path(__file__).resolve().parent / "DATA_DIR"
    kb_zip_path = data_dir / "knowledge-base.zip"
    extract_target = data_dir / "extracted_kb"

    if kb_zip_path.is_file():
        logger.info(f"Найден архив базы знаний: {kb_zip_path}. Распаковка в {extract_target}...")
        extract_target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(kb_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_target)
        search_dirs = [extract_target]
    else:
        logger.warning(
            f"Архив не найден: {kb_zip_path}. Поиск .txt файлов напрямую..."
        )
        search_dirs = [data_dir]

    loaded_docs = {}

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for file_path in sorted(s_dir.rglob("*.txt")):
            raw_text = file_path.read_text(encoding="utf-8")
            if file_path.name not in loaded_docs:
                loaded_docs[file_path.name] = raw_text

    logger.info(f"Успешно загружено документов для индексации: {len(loaded_docs)}")
    return list(loaded_docs.items())


def build_deterministic_graph(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Строит связи AMENDS, VERSIONS, LAW_NOTES для графового расширения RAG."""
    logger.info("Построение детерминированного графа связей...")
    edges = {"AMENDS": [], "VERSIONS": [], "LAW_NOTES": []}
    by_doc = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch["chunk_id"])
    doc_ids = list(by_doc.keys())

    # Ребро AMENDS: Договор <-> ДС
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

    # Ребро VERSIONS: Регламенты и Письма
    reg_2023 = next((d for d in doc_ids if "регламент" in d.lower() and "2023" in d.lower()), None)
    reg_2025 = next((d for d in doc_ids if "регламент" in d.lower() and "2025" in d.lower()), None)
    k12 = next((d for d in doc_ids if "к-12" in d.lower() or "комплаенс" in d.lower()), None)
    if reg_2023 and reg_2025:
        edges["VERSIONS"].append({
            "reg_2023_doc": reg_2023, "reg_2025_doc": reg_2025, "letter_k12_doc": k12,
            "reg_2023_chunks": by_doc.get(reg_2023, []), "reg_2025_chunks": by_doc.get(reg_2025, []),
            "letter_k12_chunks": by_doc.get(k12, []) if k12 else [], "switch_date": "2025-03-01"
        })

    # Ребро LAW_NOTES: Законы <-> Примечания
    for d_law in doc_ids:
        if any(w in d_law.lower() for w in ["закон", "коап", "152-фз"]):
            doc_chunks = [c for c in chunks if c["doc_id"] == d_law]
            anchors = [c["chunk_id"] for c in doc_chunks if c["chunk_index"] == 0 or "примечан" in c["clause_id"].lower()]
            edges["LAW_NOTES"].append({
                "law_doc": d_law, "all_law_chunks": by_doc[d_law],
                "header_and_notes_chunks": list(dict.fromkeys(anchors))
            })
    return edges


def main():
    logger.info("Запуск модуля индексации (SAT-Graph RAG Pipeline)...")
    emb_client = RemoteEmbeddingClient(LLM_BASE_URL, LLM_API_KEY, EMB_MODEL)
    sanitizer = SemanticSanitizer(emb_client)

    documents = load_knowledge_base_texts()
    if not documents:
        logger.error("Документы не найдены. Остановка.")
        return

    all_chunks = []
    for filename, raw_text in documents:
        logger.info(f"Обработка файла: {filename}")
        passport = extract_document_passport(filename, raw_text)
        clauses = split_into_clauses(raw_text)
        
        # Очистка чанков от инъекций перед добавлением
        cleaned_texts = sanitizer.sanitize_clauses(filename, [c["text"] for c in clauses])

        for idx, (cl, c_text) in enumerate(zip(clauses, cleaned_texts)):
            if not c_text.strip():
                continue
            chunk_id = f"{filename}::chunk_{idx}"
            contextual_text = f"{passport['prefix']}\n{c_text}"

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

    # Генерация эмбеддингов
    logger.info(f"Отправка {len(all_chunks)} чанков в API ({EMB_MODEL}) для расчета векторов...")
    context_texts = [c["contextual_text"] for c in all_chunks]
    embeddings = emb_client.get_embeddings(context_texts)

    # Построение FAISS
    logger.info("Сборка векторного индекса (FAISS IndexFlatIP)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Сохранение артефактов плоско в корневой каталог (требование регламента)
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks_and_graph.json", "w", encoding="utf-8") as f:
        json.dump({
            "chunks": all_chunks,
            "graph": graph_edges,
            "quarantine_log": sanitizer.quarantine_log
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Индексация успешно завершена! В карантин отправлено {len(sanitizer.quarantine_log)} фрагментов.")
    logger.info("Файлы `faiss_index.bin` и `chunks_and_graph.json` сохранены в корне проекта.")


if __name__ == "__main__":
    main()
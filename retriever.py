import os
import re
import time
import json
import faiss
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

# Конфигурация прокси тестирующей системы (согласно п. 6-7 Технического регламента)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "hackathon_key"))
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")


class LegalRetriever:
    def __init__(self, index_dir: str = "."):
        json_path = os.path.join(index_dir, "chunks_and_graph.json")
        faiss_path = os.path.join(index_dir, "faiss_index.bin")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks: List[Dict[str, Any]] = data["chunks"]
        self.graph: Dict[str, Any] = data["graph"]
        self.chunk_map: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in self.chunks}

        self.faiss_index = faiss.read_index(faiss_path)

        # Инициализация лексического индекса BM25
        tokenized_corpus = [self._tokenize(c["contextual_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.emb_endpoint = f"{LLM_BASE_URL}/embeddings"
        self.rerank_endpoint = f"{LLM_BASE_URL}/rerank"
        self.headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Токенизация с сохранением юридических идентификаторов (номера договоров, статьи, ИНН, суммы)."""
        text_lower = text.lower().replace("ё", "е")
        tokens = re.findall(r"[а-яa-z0-9]+(?:[-/.][а-яa-z0-9]+)*", text_lower)
        expanded = []
        for t in tokens:
            expanded.append(t)
            if "/" in t or "-" in t:
                expanded.append(t.replace("/", "-"))
                expanded.extend(re.split(r"[-/]", t))
        return expanded

    def _post_with_retry(self, url: str, payload: Dict[str, Any], max_retries: int = 3, timeout: int = 20) -> Dict[str, Any]:
        """Запрос к API платформы с защитой от 429 / 5xx и сетевых тайм-аутов."""
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return {}

    def _get_embedding(self, text: str) -> np.ndarray:
        data = self._post_with_retry(
            self.emb_endpoint,
            payload={"model": EMB_MODEL, "input": [text]}
        )
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm if norm > 0 else 1e-9)).reshape(1, -1)

    def retrieve(
        self,
        query: str,
        target_date: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: int = 12,
        final_k: int = 6,
        min_rerank_score: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Боевой метод гибридного поиска и графового расширения:
        1. Топ-12 по FAISS (BAAI/bge-m3) + Топ-12 по BM25.
        2. Слияние через Reciprocal Rank Fusion (RRF).
        3. Графовое расширение (AMENDS, VERSIONS, LAW_NOTES).
        4. Реранкинг через Qwen3 Reranker 8B + отсечение шума по порогу CRAG (min_rerank_score).
        """
        # 1. Векторный поиск (FAISS)
        q_emb = self._get_embedding(query)
        _, faiss_indices = self.faiss_index.search(q_emb, top_k)
        dense_hits = [
            self.chunks[idx]["chunk_id"]
            for idx in faiss_indices[0]
            if 0 <= idx < len(self.chunks)
        ]

        # 2. Лексический поиск (BM25)
        lexical_query = query
        if entities:
            lexical_query += " " + " ".join(entities)
        bm_tokens = self._tokenize(lexical_query)
        bm_scores = self.bm25.get_scores(bm_tokens)
        bm_top = np.argsort(bm_scores)[::-1][:top_k]
        sparse_hits = [
            self.chunks[idx]["chunk_id"]
            for idx in bm_top
            if bm_scores[idx] > 0
        ]

        # 3. Слияние списков через Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores: Dict[str, float] = {}
        k_rrf = 60
        for rank, cid in enumerate(dense_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, cid in enumerate(sparse_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)

        seed_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        if not seed_ids:
            return []

        # 4. Графовое расширение (Graph Expansion) по связям из chunks_and_graph.json
        expanded_ids = list(seed_ids)
        active_docs = {self.chunk_map[cid]["doc_id"] for cid in seed_ids if cid in self.chunk_map}

        # Ребро AMENDS (основной договор <-> доп. соглашения)
        for edge in self.graph.get("AMENDS", []):
            if edge["source_doc"] in active_docs:
                for cid in edge["target_chunks"]:
                    if cid not in expanded_ids:
                        expanded_ids.append(cid)
            elif edge["target_doc"] in active_docs:
                for cid in edge["source_chunks"][:2]:
                    if cid not in expanded_ids:
                        expanded_ids.append(cid)

        # Ребро VERSIONS (редакции регламентов и связанные письма по target_date)
        for v_edge in self.graph.get("VERSIONS", []):
            v_docs = {v_edge.get("reg_2023_doc"), v_edge.get("reg_2025_doc"), v_edge.get("letter_k12_doc")}
            if active_docs.intersection(v_docs):
                for cid in v_edge.get("letter_k12_chunks", []):
                    if cid not in expanded_ids:
                        expanded_ids.append(cid)

                switch_date = v_edge.get("switch_date", "2025-03-01")
                if target_date and target_date < switch_date:
                    for cid in v_edge.get("reg_2023_chunks", []):
                        if cid not in expanded_ids:
                            expanded_ids.append(cid)
                else:
                    for cid in v_edge.get("reg_2025_chunks", [])[:4]:
                        if cid not in expanded_ids:
                            expanded_ids.append(cid)

        # Ребро LAW_NOTES (статьи законов <-> шапка с датами редакций и примечания)
        for l_edge in self.graph.get("LAW_NOTES", []):
            if l_edge["law_doc"] in active_docs:
                for cid in l_edge["header_and_notes_chunks"]:
                    if cid not in expanded_ids:
                        expanded_ids.append(cid)

        # Соседний чанк с информацией о дате изменения части закона
        for cid in list(expanded_ids):
            ch = self.chunk_map.get(cid)
            if ch and any(w in ch["doc_id"].lower() for w in ["закон", "коап", "фз", "нк_", "тк_", "гк_"]):
                if ch["chunk_index"] > 0:
                    prev_id = f"{ch['doc_id']}::chunk_{ch['chunk_index'] - 1}"
                    if prev_id in self.chunk_map and prev_id not in expanded_ids:
                        expanded_ids.append(prev_id)

        # Исключаем короткие технические чанки-заголовки разделов ("1. Предмет договора")
        candidates = []
        for cid in expanded_ids:
            ch = self.chunk_map.get(cid)
            if not ch:
                continue
            raw_txt = ch["raw_text"].strip()
            is_law_edition_note = any(w in raw_txt.lower() for w in ["изменен", "дополнен", "утратил силу"])
            if len(raw_txt) > 35 or is_law_edition_note:
                candidates.append(dict(ch))

        if not candidates:
            return []

        # 5. Вызов Qwen3 Reranker 8B через прокси организаторов
        doc_payload = [c["contextual_text"] for c in candidates]
        try:
            res_data = self._post_with_retry(
                self.rerank_endpoint,
                payload={
                    "model": RERANKER_MODEL,
                    "query": query,
                    "documents": doc_payload,
                    "top_n": len(doc_payload)
                }
            )
            results = res_data.get("results", res_data.get("data", []))
            score_map = {
                item["index"]: float(item.get("relevance_score", item.get("score", 0.0)))
                for item in results
            }
            scores = [score_map.get(i, 0.0) for i in range(len(candidates))]
            is_reranker_active = True
        except Exception:
            # На случай, если прокси реранкера недоступен
            scores = [rrf_scores.get(c["chunk_id"], 0.01) for c in candidates]
            is_reranker_active = False

        for ch, sc in zip(candidates, scores):
            ch["rerank_score"] = round(float(sc), 4)
            ch["spotlighted_xml"] = (
                f'<document_evidence id="{ch["chunk_id"]}" source="{ch["doc_id"]}" '
                f'clause="{ch["clause_id"]}" status="{ch["status"]}">\n'
                f'{ch["contextual_text"]}\n'
                f'</document_evidence>'
            )

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 6. Порог достаточности контекста CRAG (отсечение вопросов вне базы)
        if is_reranker_active:
            if candidates[0]["rerank_score"] < min_rerank_score:
                return []
            candidates = [c for c in candidates if c["rerank_score"] >= min_rerank_score * 0.5]

        return candidates[:final_k]
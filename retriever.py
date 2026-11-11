import os
import re
import json
import faiss
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


class LegalGraphRetriever:
    def __init__(
        self,
        json_path: str = "chunks_and_graph.json",
        faiss_path: str = "faiss_index.bin"
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks: List[Dict[str, Any]] = data["chunks"]
        self.chunk_map: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in self.chunks}
        self.graph: Dict[str, Any] = data["graph"]

        self.faiss_index = faiss.read_index(faiss_path)

        # Конфигурация прокси из переменных окружения платформы
        self.base_url = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "hackathon_key"))
        self.emb_model = os.getenv("EMB_MODEL", "BAAI/bge-m3")
        self.reranker_model = os.getenv("RERANKER_MODEL", "Qwen3-Reranker-8B")

        # Инициализация BM25
        tokenized_corpus = [self._tokenize_legal(c["contextual_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize_legal(text: str) -> List[str]:
        text_lower = text.lower().replace("ё", "е")
        tokens = re.findall(r"[а-яa-z0-9]+(?:[-/.][а-яa-z0-9]+)*", text_lower)
        expanded = []
        for t in tokens:
            expanded.append(t)
            if "/" in t or "-" in t:
                expanded.append(t.replace("/", "-"))
                expanded.extend(re.split(r"[-/]", t))
        return expanded

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Получение вектора запроса через API."""
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.emb_model, "input": [query]},
            timeout=20
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm if norm > 0 else 1e-9)).reshape(1, -1)

    def retrieve(
        self,
        standalone_query: str,
        target_date: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: int = 12,
        final_top_k: int = 6
    ) -> List[Dict[str, Any]]:
        # 1. Параллельный поиск (FAISS + BM25)
        # Канал Dense (FAISS)
        q_emb = self._get_query_embedding(standalone_query)
        _, faiss_indices = self.faiss_index.search(q_emb, top_k)
        dense_hits = [self.chunks[idx]["chunk_id"] for idx in faiss_indices[0] if 0 <= idx < len(self.chunks)]

        # Канал Sparse (BM25)
        lexical_query = standalone_query
        if entities:
            lexical_query += " " + " ".join(entities)
        bm_tokens = self._tokenize_legal(lexical_query)
        bm_scores = self.bm25.get_scores(bm_tokens)
        bm_top = np.argsort(bm_scores)[::-1][:top_k]
        sparse_hits = [self.chunks[idx]["chunk_id"] for idx in bm_top if bm_scores[idx] > 0]

        # 2. Слияние через RRF (k=60)
        rrf: Dict[str, float] = {}
        for r, cid in enumerate(dense_hits):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + r + 1)
        for r, cid in enumerate(sparse_hits):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + r + 1)

        sorted_seed = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]

        # 3. Детерминированное графовое расширение
        expanded_ids = self._graph_expand(sorted_seed, target_date=target_date)

        # 4. Реранкинг через API Qwen3 Reranker 8B
        candidates = [self.chunk_map[cid] for cid in expanded_ids if cid in self.chunk_map]
        scored_chunks = self._rerank(standalone_query, candidates, fallback_rrf=rrf)

        return scored_chunks[:final_top_k]

    def _graph_expand(self, seed_ids: List[str], target_date: Optional[str] = None) -> List[str]:
        pool = list(seed_ids)
        active_docs = {self.chunk_map[cid]["doc_id"] for cid in seed_ids if cid in self.chunk_map}

        # Ребро AMENDS (Договор <-> ДС)
        for edge in self.graph.get("AMENDS", []):
            if edge["source_doc"] in active_docs:
                for cid in edge["target_chunks"]:
                    if cid not in pool:
                        pool.append(cid)
            elif edge["target_doc"] in active_docs:
                for cid in edge["source_chunks"][:2]:
                    if cid not in pool:
                        pool.append(cid)

        # Ребро VERSIONS (Регламенты и Письмо К-12)
        for v in self.graph.get("VERSIONS", []):
            v_docs = {v.get("reg_2023_doc"), v.get("reg_2025_doc"), v.get("letter_k12_doc")}
            if active_docs.intersection(v_docs):
                for cid in v.get("letter_k12_chunks", []):
                    if cid not in pool:
                        pool.append(cid)
                if target_date and target_date < v.get("switch_date", "2025-03-01"):
                    for cid in v.get("reg_2023_chunks", []):
                        if cid not in pool:
                            pool.append(cid)
                else:
                    for cid in v.get("reg_2025_chunks", [])[:4]:
                        if cid not in pool:
                            pool.append(cid)

        # Ребро LAW_NOTES (Статья <-> Примечания)
        for l_edge in self.graph.get("LAW_NOTES", []):
            if l_edge["law_doc"] in active_docs:
                for cid in l_edge["header_and_notes_chunks"]:
                    if cid not in pool:
                        pool.append(cid)

        return pool

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        fallback_rrf: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        doc_payload = [c["contextual_text"] for c in candidates]
        try:
            # Вызов Qwen3 Reranker 8B через прокси
            res = requests.post(
                f"{self.base_url}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.reranker_model,
                    "query": query,
                    "documents": doc_payload,
                    "top_n": len(doc_payload)
                },
                timeout=15
            )
            res.raise_for_status()
            results = res.json().get("results", res.json().get("data", []))
            score_map = {item["index"]: float(item.get("relevance_score", item.get("score", 0.0))) for item in results}
            scores = [score_map.get(i, 0.0) for i in range(len(candidates))]
        except Exception:
            scores = [fallback_rrf.get(c["chunk_id"], 0.01) for c in candidates]

        output = []
        for ch, sc in zip(candidates, scores):
            item = dict(ch)
            item["rerank_score"] = round(sc, 4)
            # Spotlighting обертка для LLM
            item["spotlighted_xml"] = (
                f'<document_evidence id="{ch["chunk_id"]}" source="{ch["doc_id"]}" '
                f'clause="{ch["clause_id"]}" status="{ch["status"]}">\n'
                f'{ch["contextual_text"]}\n'
                f'</document_evidence>'
            )
            output.append(item)

        output.sort(key=lambda x: x["rerank_score"], reverse=True)
        return output
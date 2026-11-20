"""Векторный поиск в Qdrant с фильтром доступа внутри запроса.

Безопасность:
- атрибуты доступа (уровень грифа и подразделения) хранятся в payload каждого
  фрагмента, и фильтр передаётся в Qdrant вместе с запросом (pre-filtering):
  закрытые фрагменты не участвуют в поиске ближайших соседей;
- после ответа Qdrant права проверяются повторно в коде (эшелонированная защита):
  ошибка в построении фильтра не превращается в утечку;
- фрагменты в карантине в векторный индекс не попадают, как и в BM25.

Эмбеддинги:
- ollama: модель эмбеддингов через локальную Ollama (по умолчанию bge-m3,
  многоязычная, хорошо работает с русским текстом);
- hashing: детерминированные эмбеддинги на хешировании основ слов, без модели.
  Не понимают синонимов и нужны для тестов и работы без Ollama.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request

from qdrant_client import QdrantClient, models

from .access import CLEARANCE_ORDER, User, can_read
from .index import Chunk, tokenize

COLLECTION = "chunks"
DEFAULT_MIN_SCORE = {"hashing": 0.35, "ollama": 0.5}


class EmbedderUnavailable(Exception):
    pass


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class OllamaEmbedder:
    def __init__(self, model: str | None = None, url: str | None = None):
        self.model = model or os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/") + "/api/embed"
        self.name = f"ollama:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=float(os.environ.get("OLLAMA_TIMEOUT", "300"))) as resp:
                return json.loads(resp.read())["embeddings"]
        except Exception as exc:
            # Без эмбеддингов гибридный поиск честно не работает: падаем явно, а не молча
            raise EmbedderUnavailable(
                f"Модель эмбеддингов {self.model} недоступна ({exc}). "
                f"Выполните: ollama pull {self.model}, или используйте --embedder hashing"
            ) from exc


def make_embedder(kind: str):
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"Неизвестный тип эмбеддингов: {kind}")


def access_filter(user: User) -> models.Filter:
    """Фильтр Qdrant, эквивалентный access.can_read."""
    return models.Filter(must=[
        models.FieldCondition(key="level", range=models.Range(lte=CLEARANCE_ORDER[user.clearance])),
        models.FieldCondition(key="departments", match=models.MatchAny(any=[user.department, "all"])),
    ])


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embedder, min_score: float | None = None):
        self.chunks = chunks
        self.embedder = embedder
        kind = embedder.name.split(":")[0]
        env = os.environ.get("SECURERAG_MIN_VEC_SCORE")
        self.min_score = min_score if min_score is not None else float(env) if env else DEFAULT_MIN_SCORE[kind]
        self.filter_violations = 0  # сколько раз повторная проверка отбросила фрагмент
        vectors = embedder.embed([c.title + ". " + c.text for c in chunks])
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        self.client.upsert(COLLECTION, points=[
            models.PointStruct(id=i, vector=v, payload={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "level": CLEARANCE_ORDER[c.classification],
                "departments": list(c.departments),
            })
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ])

    def _filter(self, user: User) -> models.Filter | None:
        return access_filter(user)

    def search(self, user: User, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        vec = self.embedder.embed([query])[0]
        res = self.client.query_points(COLLECTION, query=vec, query_filter=self._filter(user),
                                       limit=k, score_threshold=self.min_score)
        results = []
        for p in res.points:
            c = self.chunks[p.id]
            if not can_read(user, c.classification, list(c.departments)):
                self.filter_violations += 1
                continue
            results.append((c, float(p.score)))
        return results


    def best_score(self, user: User, query: str) -> float | None:
        """Лучшая оценка среди доступных фрагментов без порога: для настройки min_score."""
        vec = self.embedder.embed([query])[0]
        res = self.client.query_points(COLLECTION, query=vec, query_filter=self._filter(user), limit=1)
        return round(float(res.points[0].score), 3) if res.points else None


def rrf(rankings: list[list[Chunk]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: слияние ранжирований без калибровки шкал оценок."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking, 1):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[c.chunk_id] = c
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: x[1], reverse=True)

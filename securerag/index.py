"""Загрузка документов, нарезка на фрагменты, карантин и поиск BM25.

Карантин: фрагменты с признаками инъекции (косвенная инъекция через документ,
OWASP LLM01 и LLM04) исключаются из индекса на этапе загрузки и журналируются.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer
import yaml
from rank_bm25 import BM25Okapi

from .access import User, can_read
from .guardrails import detect_injection

_STEMMER = snowballstemmer.stemmer("russian")
_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_STOP = {"и", "в", "во", "на", "по", "с", "со", "к", "о", "об", "от", "до", "для", "из", "за",
         "не", "а", "но", "или", "что", "как", "это", "ли", "же", "у", "при", "то", "мне",
         "какой", "какая", "какие", "каков", "сколько", "есть"}


def tokenize(text: str) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return _STEMMER.stemWords([w for w in words if w not in _STOP])


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    classification: str
    departments: tuple[str, ...]
    text: str


def parse_doc(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    _, front, body = raw.split("---", 2)
    return yaml.safe_load(front), body.strip()


def load_chunks(docs_dir: Path) -> tuple[list[Chunk], list[dict]]:
    """Возвращает допущенные фрагменты и список помещённых в карантин."""
    chunks, quarantined = [], []
    for path in sorted(docs_dir.glob("*.md")):
        meta, body = parse_doc(path)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs, 1):
            chunk = Chunk(
                chunk_id=f"{meta['id']}#{i}",
                doc_id=meta["id"],
                title=meta["title"],
                classification=meta["classification"],
                departments=tuple(meta["departments"]),
                text=para,
            )
            hits = detect_injection(para)
            if hits:
                quarantined.append({"chunk_id": chunk.chunk_id, "patterns": hits})
                continue
            chunks.append(chunk)
    return chunks, quarantined


class SecureIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in chunks])

    def search(self, user: User, query: str, k: int = 3, min_score: float = 1.0,
               min_coverage: float = 0.4) -> list[tuple[Chunk, float]]:
        """Поиск только среди фрагментов, доступных пользователю (фильтр до ранжирования).

        min_coverage: доля значимых слов запроса, которые должны встретиться во фрагменте.
        Отсекает слабо связанные фрагменты, чтобы на вопрос о закрытом документе
        не подставлялся случайный «похожий» ответ из открытых.
        """
        permitted = [i for i, c in enumerate(self.chunks) if can_read(user, c.classification, list(c.departments))]
        q_tokens = set(tokenize(query))
        if not permitted or not q_tokens:
            return []
        scores = self.bm25.get_scores(list(q_tokens))
        results = []
        for i in permitted:
            c = self.chunks[i]
            coverage = len(q_tokens & set(tokenize(c.title + " " + c.text))) / len(q_tokens)
            if scores[i] >= min_score and coverage >= min_coverage:
                results.append((c, float(scores[i])))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

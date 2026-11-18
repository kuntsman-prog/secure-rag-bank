"""Конвейер обработки запроса и аудит-журнал.

Порядок шагов:
1. входной фильтр (инъекции, длина);
2. поиск только по доступным пользователю фрагментам;
3. единый ответ «нет данных», если доступного контекста нет
   (не раскрываем, что закрытый документ существует);
4. генерация (локальная модель или экстрактивный режим);
5. выходной фильтр (ссылки только на выданные документы) и маскирование ПДн;
6. запись в аудит-журнал (запрос в журнале тоже маскируется).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .access import UnknownUser, User, load_users
from .guardrails import check_input, check_output, mask_pii
from .index import SecureIndex, load_chunks
from .llm import NO_ANSWER, generate

ROOT = Path(__file__).resolve().parent.parent

BLOCKED_INPUT = "Запрос отклонён политикой безопасности ассистента."
BLOCKED_OUTPUT = "Ответ не прошёл проверку и не может быть показан. Обратитесь к документам напрямую."


@dataclass
class Answer:
    decision: str  # answered | no_context | blocked_input | blocked_output
    text: str
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    backend: str = ""


class SecureRAG:
    def __init__(self, docs_dir: Path | None = None, users_file: Path | None = None,
                 log_file: Path | None = None, backend: str = "extractive"):
        self.users = load_users(users_file or ROOT / "config" / "users.yaml")
        chunks, self.quarantined = load_chunks(docs_dir or ROOT / "data" / "docs")
        self.index = SecureIndex(chunks)
        self.log_file = log_file or ROOT / "logs" / "audit.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        for q in self.quarantined:
            self._audit({"event": "chunk_quarantined", **q})

    def user(self, name: str) -> User:
        if name not in self.users:
            raise UnknownUser(name)
        return self.users[name]

    def ask(self, user_name: str, question: str) -> Answer:
        started = time.perf_counter()
        user = self.user(user_name)
        result = self._ask(user, question)
        masked_q, _ = mask_pii(question)
        self._audit({
            "event": "query",
            "user": user.name,
            "role": user.role,
            "question_masked": masked_q,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest()[:16],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **asdict(result),
        })
        return result

    def _ask(self, user: User, question: str) -> Answer:
        guard = check_input(question)
        if not guard.allowed:
            return Answer("blocked_input", BLOCKED_INPUT, reasons=guard.reasons)

        hits = self.index.search(user, question)
        if not hits:
            return Answer("no_context", NO_ANSWER)

        chunks = [c for c, _ in hits]
        text, backend = generate(question, chunks, self.backend)
        if text.startswith(NO_ANSWER[:30]):
            return Answer("no_context", NO_ANSWER, backend=backend)

        sources = sorted({c.doc_id for c in chunks})
        out = check_output(text, set(sources))
        if not out.allowed:
            return Answer("blocked_output", BLOCKED_OUTPUT, reasons=out.reasons, backend=backend)

        text, pii = mask_pii(text)
        reasons = [f"pii_masked:{k}" for k in pii]
        return Answer("answered", text, sources=sources, reasons=reasons, backend=backend)

    def _audit(self, record: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

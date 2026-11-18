"""HTTP API.

Запуск: uvicorn securerag.app:app --port 8000

Аутентификация в демо упрощена: пользователь передаётся в заголовке X-User.
В промышленном варианте здесь проверяется OIDC-токен (например, Keycloak),
а роль, подразделение и уровень допуска берутся из его атрибутов.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .access import UnknownUser
from .pipeline import SecureRAG

RATE_LIMIT = 20  # запросов в минуту на пользователя (OWASP LLM10)

app = FastAPI(title="Secure RAG")
rag = SecureRAG(backend=os.environ.get("RAG_BACKEND", "extractive"))
_hits: dict[str, deque] = defaultdict(deque)


class Question(BaseModel):
    question: str


def _rate_limited(user: str) -> bool:
    now, window = time.time(), _hits[user]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return True
    window.append(now)
    return False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(rag.index.chunks), "quarantined": len(rag.quarantined)}


@app.post("/ask")
def ask(body: Question, x_user: str = Header(...)) -> dict:
    if _rate_limited(x_user):
        raise HTTPException(429, "Слишком много запросов")
    try:
        ans = rag.ask(x_user, body.question)
    except UnknownUser:
        raise HTTPException(401, "Неизвестный пользователь")
    return {"decision": ans.decision, "answer": ans.text, "sources": ans.sources, "reasons": ans.reasons}

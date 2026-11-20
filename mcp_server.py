"""MCP-сервер: доступ ИИ-агента к регламентам через Model Context Protocol.

Запуск (stdio):
  SECURERAG_USER=ivanov python -m securerag.mcp_server
  Windows: set SECURERAG_USER=ivanov && python -m securerag.mcp_server

Принципы безопасности агентного доступа:
1. Личность пользователя задаётся при запуске сервера (в промышленном варианте
   берётся из OAuth-токена HTTP-транспорта), а не параметром инструмента.
   Агент не может «представиться» другим сотрудником, даже если его об этом
   попросит внедрённая инструкция (confused deputy, OWASP LLM06).
2. Только чтение: инструменты не меняют данные и не вызывают внешние системы.
3. Описания инструментов статичны и не содержат данных из документов:
   содержимое документов не может подменить инструкции для агента (tool poisoning).
4. Каждый вызов проходит тот же конвейер, что и прямой запрос: входной фильтр,
   ABAC до поиска, выходной фильтр, маскирование ПДн; в аудит-журнал пишется
   канал «mcp».
5. Ограничение частоты вызовов (OWASP LLM10).
"""
from __future__ import annotations

import os
import time
from collections import deque

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .pipeline import SecureRAG

RATE_LIMIT = 20  # вызовов в минуту

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def build_server(user: str, rag: SecureRAG | None = None, rate_limit: int = RATE_LIMIT) -> MCPServer:
    """Создаёт MCP-сервер, привязанный к одному пользователю."""
    rag = rag or SecureRAG(
        backend=os.environ.get("SECURERAG_BACKEND", "extractive"),
        retrieval=os.environ.get("SECURERAG_RETRIEVAL", "bm25"),
        embedder=os.environ.get("SECURERAG_EMBEDDER", "ollama"),
    )
    rag.user(user)  # неизвестный пользователь: ошибка при запуске, а не при вызове
    calls: deque = deque()

    def check_rate() -> None:
        now = time.time()
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= rate_limit:
            raise ToolError("Слишком много запросов, повторите через минуту")
        calls.append(now)

    server = MCPServer(
        name="secure-rag-bank",
        instructions=(
            "Справочник по внутренним регламентам банка. Отвечает только по документам, "
            "доступным текущему пользователю, и указывает источники. Только чтение."
        ),
    )

    @server.tool(
        name="ask_regulations",
        title="Вопрос по регламентам банка",
        description=(
            "Ответ на вопрос по внутренним регламентам банка с учётом прав текущего пользователя. "
            "Возвращает решение (answered, no_context, blocked_input, blocked_output), текст ответа "
            "и идентификаторы документов-источников. Если данных нет или они недоступны, "
            "ответ одинаковый: «В доступных вам документах ответа нет»."
        ),
        annotations=READ_ONLY,
    )
    def ask_regulations(question: str) -> dict:
        check_rate()
        ans = rag.ask(user, question, channel="mcp")
        return {"decision": ans.decision, "answer": ans.text, "sources": ans.sources}

    @server.tool(
        name="list_my_documents",
        title="Доступные документы",
        description="Список регламентов, доступных текущему пользователю: идентификатор и название, без содержимого.",
        annotations=READ_ONLY,
    )
    def list_my_documents() -> dict:
        check_rate()
        return {"documents": rag.readable_documents(user)}

    return server


def main() -> None:
    user = os.environ.get("SECURERAG_USER")
    if not user:
        raise SystemExit("Задайте пользователя переменной SECURERAG_USER, например ivanov")
    build_server(user).run("stdio")


if __name__ == "__main__":
    main()

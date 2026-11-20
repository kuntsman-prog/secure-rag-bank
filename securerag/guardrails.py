"""Защитные фильтры на входе, в документах и на выходе.

Эвристики на регулярных выражениях: прозрачны, проверяемы и не требуют
внешних сервисов. В промышленном варианте их дополняют классификатором
(например, отдельной моделью-судьёй), но базовый слой правил остаётся.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Признаки прямой и косвенной инъекции промпта, попытки обойти ограничения
INJECTION_PATTERNS = [
    r"игнорируй\s+(все\s+)?(предыдущие|прошлые|данные)\s+(инструкции|указания|правила)",
    r"забудь\s+(все\s+)?(инструкции|правила|ограничения)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+",
    r"(покажи|выведи|раскрой|reveal|print|show)\s+(свой\s+|твой\s+|the\s+)?(системн\w*\s+(промпт|инструкци\w*)|system\s+prompt)",
    r"ты\s+теперь\s+",
    r"(you\s+are\s+now|act\s+as)\s+",
    r"режим\s+(разработчика|без\s+ограничений)",
    r"\bDAN\b",
    r"без\s+(каких-либо\s+)?ограничений",
    r"(обойди|отключи)\s+(фильтр|ограничени|защит)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Персональные данные и секреты, которые маскируются в ответах и в журнале
PII_PATTERNS = {
    "PHONE": r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
    "EMAIL": r"[\w.\-]+@[\w\-]+\.[\w.\-]+",
    "CARD": r"\b(?:\d[ \-]?){15}\d\b",
    "PASSPORT": r"\b\d{2}\s?\d{2}\s?\d{6}\b",
    "SECRET": r"(?i)(пароль|password|api[_\-]?key|token)\s*[:=]\s*\S+",
}
_PII_RE = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

MAX_QUESTION_LEN = 1000


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJECTION_RE if p.search(text)]


def check_input(question: str) -> GuardResult:
    """Входной фильтр: длина запроса и признаки инъекции (OWASP LLM01, LLM10)."""
    reasons = []
    if not question.strip():
        reasons.append("empty_question")
    if len(question) > MAX_QUESTION_LEN:
        reasons.append("question_too_long")
    if detect_injection(question):
        reasons.append("prompt_injection")
    return GuardResult(allowed=not reasons, reasons=reasons)


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Маскирует ПДн и секреты. Возвращает текст и список найденных типов."""
    found = []
    for kind, rx in _PII_RE.items():
        if rx.search(text):
            found.append(kind)
            text = rx.sub(f"[{kind} скрыт]", text)
    return text, found


CITATION_RE = re.compile(r"\[(DOC-\d{3})\]")


def check_output(answer: str, allowed_doc_ids: set[str]) -> GuardResult:
    """Выходной фильтр: ответ ссылается только на документы из выдачи (OWASP LLM05, LLM09)."""
    reasons = []
    cited = set(CITATION_RE.findall(answer))
    if not cited:
        reasons.append("no_citations")
    if cited - allowed_doc_ids:
        reasons.append("citation_outside_context")
    if detect_injection(answer):
        reasons.append("injection_echo")
    return GuardResult(allowed=not reasons, reasons=reasons)

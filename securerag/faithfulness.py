"""Оценка достоверности ответа: опирается ли ответ на выданный контекст.

Проверки детерминированные и воспроизводимые, без модели-судьи:
- утверждения: каждое предложение ответа должно опираться на контекст
  (доля значимых слов утверждения, найденных в контексте, не ниже порога);
- числа: каждое число в ответе должно встречаться в контексте
  (выдуманные суммы, сроки и проценты для проверки документов критичны);
- ссылки: в ответе есть хотя бы одна ссылка, и все ссылки ведут на выданные документы;
- ожидаемые факты: ответ содержит эталонный факт из набора проверок;
- корректный отказ: на вопрос, ответа на который в доступных документах нет,
  ассистент отвечает «данных нет», а не придумывает.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import tokenize

CLAIM_SUPPORT_THRESHOLD = 0.7
MIN_CLAIM_TOKENS = 3

_CITATION_RE = re.compile(r"\[?\bDOC-\d+(?:#\d+)?\]?")
_NUMBER_RE = re.compile(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?")
_HEADER_RE = re.compile(r"^по доступным вам документам:?\s*", re.IGNORECASE)


def normalize(text: str) -> str:
    """Нижний регистр, ё как е, пробелы внутри чисел убраны, десятичная точка как запятая."""
    t = text.lower().replace("ё", "е").replace(" ", " ")
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)
    return re.sub(r"(?<=\d)\.(?=\d)", ",", t)


def numbers(text: str) -> set[str]:
    return {normalize(n) for n in _NUMBER_RE.findall(_CITATION_RE.sub(" ", text))}


def claims(answer: str) -> list[str]:
    """Разбивает ответ на утверждения: строки и предложения без ссылок, маркеров списка и вопросов."""
    text = _HEADER_RE.sub("", answer.strip())
    result = []
    for line in text.splitlines():
        line = _CITATION_RE.sub(" ", line).strip(" -•*\t")
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip()
            # Вопрос ничего не утверждает: модель может повторить его в начале ответа
            if sent.endswith("?"):
                continue
            if len(tokenize(sent)) >= MIN_CLAIM_TOKENS:
                result.append(sent)
    return result


def claim_support(claim: str, context: str) -> float:
    words = tokenize(claim)
    if not words:
        return 1.0
    ctx = set(tokenize(context))
    return sum(w in ctx for w in words) / len(words)


@dataclass
class Check:
    claims_total: int = 0
    claims_supported: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    citations_ok: bool = False
    missing_facts: list[str] = field(default_factory=list)


def check_answer(answer: str, context: str, sources: set[str], expect_facts: list[str] | None = None) -> Check:
    res = Check()
    for c in claims(answer):
        res.claims_total += 1
        if claim_support(c, context) >= CLAIM_SUPPORT_THRESHOLD:
            res.claims_supported += 1
        else:
            res.unsupported_claims.append(c)
    ctx_numbers = numbers(context)
    res.unsupported_numbers = sorted(n for n in numbers(answer) if n not in ctx_numbers)
    cited = set(re.findall(r"DOC-\d+", answer))
    res.citations_ok = bool(cited) and cited <= sources
    norm = normalize(answer)
    res.missing_facts = [f for f in (expect_facts or []) if normalize(f) not in norm]
    return res

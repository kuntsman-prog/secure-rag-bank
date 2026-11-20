"""Контроль доступа на основе атрибутов (ABAC).

Ключевое правило: фильтрация выполняется ДО поиска. Фрагменты, к которым у
пользователя нет доступа, не участвуют в ранжировании и никогда не попадают
в контекст модели. Это защищает от утечки через RAG (OWASP LLM02, LLM08).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CLEARANCE_ORDER = {"public": 0, "internal": 1, "confidential": 2}


@dataclass(frozen=True)
class User:
    name: str
    role: str
    department: str
    clearance: str


class UnknownUser(Exception):
    pass


def load_users(path: Path) -> dict[str, User]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["users"]
    return {
        name: User(name=name, role=a["role"], department=a["department"], clearance=a["clearance"])
        for name, a in raw.items()
    }


def can_read(user: User, classification: str, departments: list[str]) -> bool:
    """Разрешён ли пользователю документ с данной меткой и списком подразделений."""
    if CLEARANCE_ORDER[user.clearance] < CLEARANCE_ORDER[classification]:
        return False
    return "all" in departments or user.department in departments

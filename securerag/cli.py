"""Командная строка.

Примеры:
  python -m securerag.cli ask --user ivanov "Какой срок ответа на обращение клиента?"
  python -m securerag.cli ask --user ivanov --backend ollama "..."
  python -m securerag.cli eval
  python -m securerag.cli quarantine
"""
from __future__ import annotations

import argparse
import json
import sys

from .evaluate import run
from .pipeline import SecureRAG


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="securerag")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="задать вопрос от имени пользователя")
    a.add_argument("--user", required=True)
    a.add_argument("--backend", choices=["extractive", "ollama"], default="extractive")
    a.add_argument("question")
    e = sub.add_parser("eval", help="прогнать набор проверок")
    e.add_argument("--backend", choices=["extractive", "ollama"], default="extractive")
    sub.add_parser("quarantine", help="показать фрагменты в карантине")
    args = p.parse_args(argv)

    if args.cmd == "ask":
        rag = SecureRAG(backend=args.backend)
        ans = rag.ask(args.user, args.question)
        print(f"[{ans.decision}] источники: {', '.join(ans.sources) or 'нет'}  режим: {ans.backend or '-'}")
        if ans.reasons:
            print("причины:", ", ".join(ans.reasons))
        print(ans.text)
    elif args.cmd == "eval":
        report = run(SecureRAG(backend=args.backend))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["leaks"] == 0 and not report["failures"] else 1
    elif args.cmd == "quarantine":
        print(json.dumps(SecureRAG().quarantined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

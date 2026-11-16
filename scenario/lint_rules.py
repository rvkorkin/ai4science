"""Простой линтер академического русского: проверяет текст на запрещённые
обороты из словаря. Используется критиком как внешняя команда."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RULES = [
    (r"\bважно (отметить|подчеркнуть)\b", "подводка: убрать"),
    (r"\bстоит (отметить|подчеркнуть)\b", "подводка: убрать"),
    (r"\bкак известно\b", "подводка: дать ссылку или убрать"),
    (r"\bиграет (ключевую|важную) роль\b", "штамп"),
    (r"\b(мощн|элегантн|революционн|впечатляющ)\w+", "оценочное слово"),
    (r"[!]\s*$", "восклицание"),
    (r"\bдавайте\b", "обращение к читателю"),
]


def lint(text: str) -> list[dict]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("#", "```", "|", "роль:", "раунд:", "основано на:")):
            continue
        for pat, note in RULES:
            for m in re.finditer(pat, line, flags=re.IGNORECASE):
                out.append({"line": n, "text": m.group(0), "note": note})
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scenario/lint_rules.py <file>", file=sys.stderr)
        return 2
    p = Path(argv[0])
    if not p.is_file():
        print(f"нет файла {p}", file=sys.stderr)
        return 2
    res = lint(p.read_text(encoding="utf-8"))
    print(json.dumps({"file": argv[0], "findings": len(res), "items": res},
                     ensure_ascii=False, indent=1))
    return 0 if not res else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

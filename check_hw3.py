"""Самопроверка пакета ДЗ 3: python check_hw3.py (из корня своего проекта). Код 0 — можно сдавать.

Проверяется наличие и форма, а не содержание: содержание оценивает преподаватель по отчёту.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
problems, passed = [], []


def check(ok: bool, text: str) -> None:
    (passed if ok else problems).append(text)


# 1. инструмент: типы, строка документации, проверка без агента
tools = [f for f in ROOT.rglob("*_mcp.py") if not any(p.startswith(".") or p == "node_modules" for p in f.parts)]
check(bool(tools), "есть файл инструмента *_mcp.py")
for f in tools:
    tree = ast.parse(f.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any("tool" in ast.unparse(d) for d in node.decorator_list):
            found = True
            args = [a for a in node.args.args if a.arg != "self"]
            check(all(a.annotation is not None for a in args) and node.returns is not None,
                  f"{f.name}: у инструмента {node.name} типы аргументов и результата")
            doc = ast.get_docstring(node) or ""
            check(len(doc.split()) >= 12, f"{f.name}: у инструмента {node.name} строка документации (не короче 12 слов)")
    check(found, f"{f.name}: есть функция с декоратором tool")
    run = subprocess.run([sys.executable, str(f), "--selftest"], capture_output=True, text=True, timeout=120)
    check(run.returncode == 0, f"{f.name}: --selftest завершается с кодом 0")

# 2. шина: записи обеих ролей и вердикт
bus = ROOT / "bus"
index = (bus / "INDEX.md").read_text(encoding="utf-8") if (bus / "INDEX.md").is_file() else ""
check("генератор" in index and "критик" in index, "bus/INDEX.md: есть записи генератора и критика")
reviews = sorted(bus.glob("review_*.md")) if bus.is_dir() else []
check(bool(reviews) and all(r.read_text(encoding="utf-8").startswith("ВЕРДИКТ:") for r in reviews),
      "каждая рецензия критика начинается строкой «ВЕРДИКТ:»")

# 3. журнал сценария
log = ROOT / "scenario" / "journal.log"
lines = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
stamped = [l for l in lines if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", l)]
check(len(stamped) >= 5, "scenario/journal.log: не меньше пяти записей с отметкой времени")
check(any("итог" in l for l in lines), "scenario/journal.log: есть строка с итогом сценария")

# 4. метка git
tags = subprocess.run(["git", "tag", "--list", "hw3"], cwd=ROOT, capture_output=True, text=True).stdout.split()
check("hw3" in tags, "метка git hw3 существует")

# 5. отчёт
report = next((f for f in (ROOT / "REPORT.md", ROOT / "report.md") if f.is_file()), None)
text = report.read_text(encoding="utf-8") if report else ""
check(report is not None, "есть REPORT.md")
check("tmux ls" in text, "в отчёте есть вывод tmux ls")
check("▸" in text, "в отчёте есть строка вызова инструмента из сеанса")
check(any(m in text for m in ("[порождено]", "[написано]", "[источник]")), "в отчёте есть пометки авторства")

for t in passed:
    print("  да  ", t)
for t in problems:
    print("  НЕТ ", t)
print(f"\nпройдено {len(passed)} из {len(passed) + len(problems)}")
sys.exit(1 if problems else 0)

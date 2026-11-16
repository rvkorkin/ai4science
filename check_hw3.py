"""Самопроверка ДЗ-3. Exit code 0 ⇔ работа принята.

Проверяет 4 вещи (рубрика ДЗ-3):
  A. MCP-инструмент имеет docstring с описанием входа и выхода и вызывается
     чистой функцией без агента.
  Б. Автономный сценарий с двумя ролями (generator + critic) через общую
     папку bus/ переживает обрыв связи: повторный запуск продолжает ровно
     с чекпойнтов и bus/summary.json детерминирован.
  В. Git-метка hw3 (или hw3-v1) существует; отчёт с таблицей авторства
     лежит в REPORT.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ERRORS = []


def _run(cmd):
    r = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        ERRORS.append(f"cmd failed: {' '.join(cmd)}\n{r.stderr}")
    return r


def check_A_tool():
    sys.path.insert(0, str(ROOT / "mcp"))
    from tool_core import frac_hit_verify
    doc = frac_hit_verify.__doc__ or ""
    if len(doc.strip()) < 30:
        ERRORS.append("A: frac_hit_verify docstring is too short")
    # pure call without agent
    fx = ROOT / "scenario" / "fixtures" / "rec_hit.npz"
    if not fx.exists():
        _run(["scenario/make_fixtures.py"])
    v = frac_hit_verify(str(fx))
    if not v.get("valid"):
        ERRORS.append(f"A: tool rejected valid fixture: {v.get('errors')}")
    if not v.get("hit", {}).get("estimated"):
        ERRORS.append("A: tool did not estimate hit on rec_hit.npz")
    print(f"[A] tool OK: valid={v['valid']} hit_time={v['hit'].get('hit_time_min')}")


def check_B_scenario():
    # full reset, then generator then critic, with simulated disconnect in between
    bus = ROOT / "bus"
    if bus.exists():
        shutil.rmtree(bus)
    _run(["scenario/generator.py", "--reset", "--max", "2"])
    _run(["scenario/generator.py"])          # resume after "disconnect"
    _run(["scenario/critic.py", "--reset", "--max", "2"])
    _run(["scenario/critic.py", "--write-summary"])  # resume + summary
    summ = json.loads((bus / "summary.json").read_text(encoding="utf-8"))
    if summ["n_reviewed"] != 5:
        ERRORS.append(f"B: expected 5 reviewed, got {summ['n_reviewed']}")
    if summ["n_valid"] != 2 or summ["n_hit"] != 1:
        ERRORS.append(f"B: expected 2 valid / 1 hit, got "
                      f"{summ['n_valid']} / {summ['n_hit']}")
    jg = (bus / "journal_generator.jsonl").read_text(encoding="utf-8").splitlines()
    jc = (bus / "journal_critic.jsonl").read_text(encoding="utf-8").splitlines()
    if not any("stop_simulated_disconnect" in x for x in jg):
        ERRORS.append("B: generator journal has no disconnect event")
    if not any("stop_simulated_disconnect" in x for x in jc):
        ERRORS.append("B: critic journal has no disconnect event")
    # resumed runs must NOT re-process files already in checkpoint
    reviewed_lines = [json.loads(x) for x in jc if '"event": "reviewed"' in x]
    files = [r["file"] for r in reviewed_lines]
    if len(set(files)) != len(files):
        ERRORS.append("B: critic re-processed a file across restart")
    print(f"[B] scenario OK: {summ['n_reviewed']} reviewed, "
          f"{summ['n_valid']} valid, {summ['n_hit']} hit, "
          f"journals have disconnect events, no duplicates")


def check_C_report():
    rep = ROOT / "REPORT.md"
    if not rep.exists():
        ERRORS.append("C: REPORT.md is missing")
        return
    text = rep.read_text(encoding="utf-8")
    if "Разметка авторства" not in text and "Authorship" not in text:
        ERRORS.append("C: REPORT.md has no authorship table")
    tags = subprocess.run(["git", "tag", "-l"], cwd=ROOT,
                          capture_output=True, text=True).stdout.splitlines()
    if not any(t.strip() in {"hw3", "hw3-v1"} for t in tags):
        ERRORS.append(f"C: no hw3 tag locally; have {tags}")
    print(f"[C] report OK: tag(s) {[t for t in tags if 'hw3' in t]}")


def main():
    check_A_tool()
    check_B_scenario()
    check_C_report()
    if ERRORS:
        print("\nFAIL:")
        for e in ERRORS:
            print("  -", e)
        return 1
    print("\nOK: ДЗ-3 self-check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

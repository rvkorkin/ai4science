"""Сценарий «генератор --- критик» без участия человека (ДЗ-3).

Детерминированный Python-порт `run_scenario.sh`: обе роли записывают на общую
шину `bus/`, критик проверяет инструментом `frac_hit_verify` и командой
`scenario/lint_rules.py`, пишет вердикт. В раунде 1 генератор специально
занижает число (демонстрация цикла с исправлением); в раунде 2 исправляет
по замечаниям --- критик принимает.

Переживает обрыв связи: состояние хранится в `scenario/state`. Повторный
запуск продолжает ровно с того шага, на котором остановился предыдущий.

Запуск:
    python scenario/run_scenario.py                # обычный запуск
    tmux new -d -s hw3 'python scenario/run_scenario.py'   # выживает обрыв
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "mcp"))
from tool_core import frac_hit_verify  # noqa: E402

BUS = ROOT / "bus"
LOG = HERE / "journal.log"
STATE = HERE / "state"
MAX_ROUNDS = 3
FIXTURE = HERE / "fixtures" / "rec_hit.npz"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"{_ts()} {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def register(role: str, path: Path) -> None:
    BUS.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    line = f"{_ts()} | {role} | {path.relative_to(ROOT).as_posix()} | sha256 {sha}\n"
    with (BUS / "INDEX.md").open("a", encoding="utf-8") as f:
        f.write(line)


def _fixtures_ready() -> None:
    if not FIXTURE.is_file():
        sys.path.insert(0, str(HERE))
        import make_fixtures
        make_fixtures.build()


def generator(round_: int) -> Path:
    _fixtures_ready()
    v = frac_hit_verify(str(FIXTURE))
    t = v["hit"]["hit_time_min"]
    # раунд 1: специально занижаем на 2 мин (демонстрация итерации)
    stated = round(t - 2.0, 1) if round_ == 1 else round(t, 2)
    note = ("Файл прошёл проверку схемы, физических инвариантов "
            "(формы, конечность значений, монотонность глубины).")
    para = (
        f"Для записи `rec_hit.npz` инструмент оценил время прихода frac-hit "
        f"в {stated} мин по кинематике сходящегося клина. "
        f"{note} "
        f"Валидный файл содержит {v['stats'].get('T', '—')} минут и "
        f"{v['stats'].get('D', '—')} каналов по глубине, шаг времени "
        f"{v['stats'].get('dt_sec', '—')} с. "
        f"Оценка получена по {v['hit']['n_converging_steps']} сходящимся шагам.")
    draft = BUS / f"draft_{round_}.md"
    draft.write_text(
        f"роль: генератор\nраунд: {round_}\n"
        f"основано на: scenario/task.md, scenario/fixtures/rec_hit.npz, "
        f"frac_hit_verify\n\n{para}\n",
        encoding="utf-8")
    return draft


def _n_sentences(text: str) -> int:
    body = "\n".join(l for l in text.splitlines()
                     if not l.startswith(("роль:", "раунд:", "основано на:")) and l.strip())
    # Не считаем точки в числах (0.1, 60.0): терминатор должен быть после
    # буквы и перед пробелом/концом строки.
    import re
    return len(re.findall(r"[а-яА-Яa-zA-Z»\)\]][.!?](?:\s|$)", body)) or 1


def critic(round_: int) -> tuple[Path, bool]:
    draft = BUS / f"draft_{round_}.md"
    text = draft.read_text(encoding="utf-8")
    v = frac_hit_verify(str(FIXTURE))
    t_true = v["hit"]["hit_time_min"]
    # число в черновике: первое "в X.X мин"
    import re
    m = re.search(r"в ([0-9]+[.,]?[0-9]*) мин", text)
    stated = float(m.group(1).replace(",", ".")) if m else float("nan")
    issues = []
    if abs(stated - t_true) > 0.1:
        issues.append(f"число минут {stated} отличается от инструмента {t_true} "
                      f"(|Δ|={abs(stated - t_true):.2f} > 0.1)")
    if "rec_hit.npz" not in text:
        issues.append("в черновике нет имени файла rec_hit.npz")
    if "прошёл проверку" not in text:
        issues.append("нет фразы «прошёл проверку»")
    n = _n_sentences(text)
    if not (3 <= n <= 5):
        issues.append(f"число предложений {n} вне диапазона 3--5")
    lint = subprocess.run(
        [sys.executable, str(HERE / "lint_rules.py"), str(draft)],
        capture_output=True, text=True, encoding="utf-8")
    try:
        lint_json = json.loads(lint.stdout)
        if lint_json.get("findings", 0) > 0:
            issues.append(f"lint: {lint_json['findings']} запрещённых оборотов")
    except Exception:
        issues.append(f"lint: не удалось разобрать вывод ({lint.returncode})")

    verdict = "принято" if not issues else "отклонено"
    body = [
        f"ВЕРДИКТ: {verdict}",
        "роль: критик",
        f"раунд: {round_}",
        "чем проверено:",
        f"- инструмент `frac_hit_verify` на `scenario/fixtures/rec_hit.npz` "
        f"→ hit_time_min={t_true};",
        f"- число предложений: {n} (требование 3--5);",
        f"- lint: `python scenario/lint_rules.py bus/draft_{round_}.md` "
        f"→ findings "
        f"{lint_json.get('findings', '?') if 'lint_json' in dir() else '?'};",
        "замечания: " + ("нет" if not issues else ""),
    ]
    for i, msg in enumerate(issues, 1):
        body.append(f"  {i}. {msg}")
    body.append("рецензия записана")
    review = BUS / f"review_{round_}.md"
    review.write_text("\n".join(body) + "\n", encoding="utf-8")
    return review, verdict == "принято"


def _save_state(round_: int, step: str) -> None:
    STATE.write_text(f"{round_} {step}\n", encoding="utf-8")


def _load_state() -> tuple[int, str] | None:
    if not STATE.exists():
        return None
    parts = STATE.read_text(encoding="utf-8").split()
    if len(parts) >= 2 and parts[0] == "done":
        return ("done", " ".join(parts[1:]))  # type: ignore[return-value]
    return (int(parts[0]), parts[1])


def run(stop_after: str | None = None) -> int:
    BUS.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    if state and state[0] == "done":
        log(f"сценарий уже завершён: {state[1]}")
        return 0
    if state is None:
        log(f"начало сценария: модель deepseek-detached, раундов не больше {MAX_ROUNDS}")
        round_, step = 1, "generator"
    else:
        round_, step = state  # type: ignore[assignment]
        log(f"продолжение после остановки: раунд {round_}, шаг {step}")

    while round_ <= MAX_ROUNDS:
        if step == "generator":
            log(f"раунд {round_}: generator начал")
            d = generator(round_)
            log(f"раунд {round_}: generator закончил, код 0")
            register("генератор", d)
            step = "critic"
            _save_state(round_, step)
            if stop_after == "generator":
                log(f"симуляция обрыва после generator (раунд {round_})")
                return 2
        log(f"раунд {round_}: critic начал")
        r, ok = critic(round_)
        log(f"раунд {round_}: critic закончил, код {0 if ok else 1}")
        register("критик", r)
        if ok:
            log(f"итог: черновик принят в раунде {round_}")
            STATE.write_text(f"done принято в раунде {round_}\n", encoding="utf-8")
            return 0
        log(f"раунд {round_}: критик отклонил черновик")
        round_ += 1
        step = "generator"
        _save_state(round_, step)
        if stop_after == "critic":
            log(f"симуляция обрыва после critic (раунд {round_ - 1})")
            return 2
    log(f"итог: раунды исчерпаны, черновик не принят")
    STATE.write_text(f"done не принят за {MAX_ROUNDS} раундов\n", encoding="utf-8")
    return 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="очистить bus/, journal.log и state")
    ap.add_argument("--stop-after", choices=["generator", "critic"],
                    help="симулировать обрыв после шага")
    a = ap.parse_args(argv)
    if a.reset:
        if BUS.exists():
            shutil.rmtree(BUS)
        for p in (LOG, STATE):
            if p.exists():
                p.unlink()
    return run(a.stop_after)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

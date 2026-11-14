"""Автономный сценарий пакетной верификации (ДЗ-3).

Прогоняет каждый .npz в папке через инструмент frac_hit_verify, пишет JSONL-журнал
и чекпойнт, поэтому прогон ПЕРЕЖИВАЕТ ОБРЫВ (например, разрыв связи): повторный
запуск продолжает с чекпойнта и пропускает уже обработанные файлы.

Детерминирован при фиксированных фикстурах → summary.json воспроизводится под git-меткой.

Запуск:
    python scenario/run_autonomous.py [--data DIR] [--max N] [--reset]
    --max N : обработать не более N новых файлов и выйти (симуляция обрыва связи)
    --reset : очистить чекпойнт/журнал и начать заново
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcp"))
from tool_core import frac_hit_verify  # noqa: E402

STATE = HERE / "state"
JOURNAL = HERE / "journal.jsonl"
CHECKPOINT = STATE / "checkpoint.json"
SUMMARY = HERE / "summary.json"


def _load_ckpt():
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text())["done"])
    return set()


def _save_ckpt(done):
    STATE.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps({"done": sorted(done)}, indent=2))


def _journal(entry):
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_summary(files, done):
    """Детерминированный summary (без таймстемпов) = воспроизводимый артефакт."""
    recs = []
    for p in sorted(files):
        if p.name in done:
            v = frac_hit_verify(str(p))
            recs.append({
                "file": p.name,
                "valid": v["valid"],
                "errors": v["errors"],
                "hit_estimated": v["hit"].get("estimated", False),
                "hit_time_min": v["hit"].get("hit_time_min"),
            })
    summary = {
        "n_files": len(files),
        "n_processed": len(done),
        "n_valid": sum(r["valid"] for r in recs),
        "n_hit": sum(r["hit_estimated"] for r in recs),
        "records": recs,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def run(data_dir, max_new=None):
    files = sorted(Path(data_dir).glob("*.npz"))
    done = _load_ckpt()
    processed = 0
    for p in files:
        if p.name in done:
            continue
        if max_new is not None and processed >= max_new:
            print(f"[scenario] STOP after {processed} new files (симуляция обрыва); "
                  f"{len(done)}/{len(files)} готово — перезапусти для продолжения")
            return False
        t0 = time.time()
        verdict = frac_hit_verify(str(p))
        _journal({
            "ts": datetime.now(timezone.utc).isoformat(),
            "file": p.name,
            "valid": verdict["valid"],
            "n_errors": len(verdict["errors"]),
            "n_warnings": len(verdict["warnings"]),
            "hit": verdict["hit"],
            "elapsed_s": round(time.time() - t0, 4),
        })
        done.add(p.name)
        _save_ckpt(done)  # чекпойнт ПОСЛЕ КАЖДОГО файла → устойчиво к падению
        processed += 1
        print(f"[scenario] {p.name}: valid={verdict['valid']} "
              f"hit={verdict['hit'].get('estimated')}")
    _write_summary(files, done)
    print(f"[scenario] COMPLETE: {len(done)}/{len(files)} файлов")
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "fixtures"))
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args(argv)

    if a.reset:
        for f in (JOURNAL, CHECKPOINT, SUMMARY):
            if f.exists():
                f.unlink()

    data = Path(a.data)
    if not list(data.glob("*.npz")):
        sys.path.insert(0, str(HERE))
        import make_fixtures
        make_fixtures.build()

    run(a.data, a.max)


if __name__ == "__main__":
    main(sys.argv[1:])

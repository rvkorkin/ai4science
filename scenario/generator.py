"""Role GENERATOR of the ДЗ-3 scenario.

Строит детерминированные `.npz`-записи из фикстур и выкладывает их в общий
обменник `bus/inbox/`. На каждую запись пишет строку в журнал с меткой времени.
Пройдя обрыв связи, продолжает ровно с того, что ещё не выложил
(чекпойнт `bus/state/generator.json`). Детерминирован под фиксированной меткой git.

Run:
    python scenario/generator.py [--max N] [--reset]
    --max N  : выложить не более N новых файлов (симуляция обрыва связи)
    --reset  : очистить bus/ и начать с нуля
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUS = HERE.parent / "bus"
INBOX = BUS / "inbox"
STATE = BUS / "state"
JOURNAL = BUS / "journal_generator.jsonl"
CKPT = STATE / "generator.json"


def _load():
    return set(json.loads(CKPT.read_text())["published"]) if CKPT.exists() else set()


def _save(done):
    STATE.mkdir(parents=True, exist_ok=True)
    CKPT.write_text(json.dumps({"published": sorted(done)}, indent=2))


def _log(entry):
    BUS.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _ensure_fixtures():
    fx = HERE / "fixtures"
    if not list(fx.glob("*.npz")):
        sys.path.insert(0, str(HERE))
        import make_fixtures
        make_fixtures.build()
    return sorted(fx.glob("*.npz"))


def run(max_new=None):
    INBOX.mkdir(parents=True, exist_ok=True)
    published = _load()
    produced = 0
    for src in _ensure_fixtures():
        if src.name in published:
            continue
        if max_new is not None and produced >= max_new:
            _log({"ts": datetime.now(timezone.utc).isoformat(),
                  "role": "generator", "event": "stop_simulated_disconnect",
                  "produced": produced, "total_published": len(published)})
            print(f"[gen] STOP after {produced} new (симуляция обрыва); "
                  f"{len(published)} лежит в bus/inbox")
            return False
        t0 = time.time()
        shutil.copy2(src, INBOX / src.name)
        published.add(src.name)
        _save(published)
        _log({"ts": datetime.now(timezone.utc).isoformat(),
              "role": "generator", "event": "published",
              "file": src.name, "bytes": (INBOX / src.name).stat().st_size,
              "elapsed_s": round(time.time() - t0, 4)})
        produced += 1
        print(f"[gen] published {src.name}")
    _log({"ts": datetime.now(timezone.utc).isoformat(),
          "role": "generator", "event": "complete",
          "total_published": len(published)})
    print(f"[gen] COMPLETE: {len(published)} files in bus/inbox")
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args(argv)
    if a.reset:
        for p in (JOURNAL, CKPT):
            if p.exists():
                p.unlink()
        if INBOX.exists():
            shutil.rmtree(INBOX)
        verdicts = BUS / "verdicts"
        if verdicts.exists():
            shutil.rmtree(verdicts)
    run(a.max)


if __name__ == "__main__":
    main(sys.argv[1:])

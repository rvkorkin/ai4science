"""Role CRITIC of the ДЗ-3 scenario.

Читает `bus/inbox/`, для каждой записи вызывает MCP-инструмент `frac_hit_verify`,
пишет JSON-вердикт в `bus/verdicts/` и строку в журнал. Устойчив к обрыву
связи: при повторном запуске пропускает уже разобранные файлы
(чекпойнт `bus/state/critic.json`).

Run:
    python scenario/critic.py [--max N] [--reset] [--write-summary]
    --max N          : обработать не более N новых файлов
    --reset          : очистить вердикты и чекпойнт критика
    --write-summary  : после прохода записать детерминированный bus/summary.json
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
sys.path.insert(0, str(HERE.parent / "mcp"))
from tool_core import frac_hit_verify  # noqa: E402

BUS = HERE.parent / "bus"
INBOX = BUS / "inbox"
VERDICTS = BUS / "verdicts"
STATE = BUS / "state"
JOURNAL = BUS / "journal_critic.jsonl"
CKPT = STATE / "critic.json"
SUMMARY = BUS / "summary.json"


def _load():
    return set(json.loads(CKPT.read_text())["reviewed"]) if CKPT.exists() else set()


def _save(done):
    STATE.mkdir(parents=True, exist_ok=True)
    CKPT.write_text(json.dumps({"reviewed": sorted(done)}, indent=2))


def _log(entry):
    BUS.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _summary():
    recs = []
    for vp in sorted(VERDICTS.glob("*.json")):
        v = json.loads(vp.read_text(encoding="utf-8"))
        recs.append({
            "file": v["file"], "valid": v["valid"],
            "errors": v["errors"],
            "hit_estimated": v["hit"].get("estimated", False),
            "hit_time_min": v["hit"].get("hit_time_min"),
        })
    summ = {
        "n_reviewed": len(recs),
        "n_valid": sum(r["valid"] for r in recs),
        "n_hit": sum(r["hit_estimated"] for r in recs),
        "records": recs,
    }
    SUMMARY.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    return summ


def run(max_new=None, write_summary=False):
    VERDICTS.mkdir(parents=True, exist_ok=True)
    done = _load()
    reviewed = 0
    if not INBOX.exists():
        print("[crt] bus/inbox is empty; waiting for generator")
        return False
    for p in sorted(INBOX.glob("*.npz")):
        if p.name in done:
            continue
        if max_new is not None and reviewed >= max_new:
            _log({"ts": datetime.now(timezone.utc).isoformat(),
                  "role": "critic", "event": "stop_simulated_disconnect",
                  "reviewed": reviewed, "total_done": len(done)})
            print(f"[crt] STOP after {reviewed} new (симуляция обрыва); "
                  f"{len(done)} уже разобрано")
            return False
        t0 = time.time()
        v = frac_hit_verify(str(p))
        (VERDICTS / f"{p.stem}.json").write_text(
            json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
        done.add(p.name)
        _save(done)
        _log({"ts": datetime.now(timezone.utc).isoformat(),
              "role": "critic", "event": "reviewed",
              "file": p.name, "valid": v["valid"],
              "hit": v["hit"].get("estimated", False),
              "elapsed_s": round(time.time() - t0, 4)})
        reviewed += 1
        print(f"[crt] {p.name}: valid={v['valid']} hit={v['hit'].get('estimated')}")
    if write_summary:
        s = _summary()
        _log({"ts": datetime.now(timezone.utc).isoformat(),
              "role": "critic", "event": "summary_written",
              "n_reviewed": s["n_reviewed"], "n_valid": s["n_valid"],
              "n_hit": s["n_hit"]})
    print(f"[crt] COMPLETE: {len(done)} reviewed")
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--write-summary", action="store_true")
    a = ap.parse_args(argv)
    if a.reset:
        for p in (JOURNAL, CKPT, SUMMARY):
            if p.exists():
                p.unlink()
        if VERDICTS.exists():
            shutil.rmtree(VERDICTS)
    run(a.max, a.write_summary)


if __name__ == "__main__":
    main(sys.argv[1:])

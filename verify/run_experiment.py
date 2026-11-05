"""Numerical experiment: validate every enriched DAS record and audit the split.

Plan:
  1. Resolve the data directory (FIELD_DATA_DIR from the fork .env).
  2. Load the canonical label file (core/field_strongest_hits.json).
  3. Run the schema/physical validator over all *.npz -> per-file CSV + summary JSON.
  4. Audit core/split.json: train/val stem disjointness, label coverage,
     tpk-in-range, and treatment-well group overlap.
  5. Write a combined machine-readable report to verify/results/.

Run:
    python run_experiment.py            # uses FIELD_DATA_DIR from .env
    python run_experiment.py <data_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CORE = PROJECT / "core"
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
from validate_data import validate_dir, write_reports  # noqa: E402
from test_leakage import well_group  # noqa: E402


def _read_env_field_dir() -> str | None:
    env = PROJECT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("FIELD_DATA_DIR"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _load_labels() -> dict:
    d = json.loads((CORE / "field_strongest_hits.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def audit_split(labels: dict) -> dict:
    sp = json.loads((CORE / "split.json").read_text(encoding="utf-8"))
    train, val = sp.get("train", []), sp.get("val", [])
    s_tr, s_va = set(train), set(val)
    stem_overlap = sorted(s_tr & s_va)
    missing = [s for s in train + val if s not in labels]
    tpk_bad = []
    for s in train + val:
        rec = labels.get(s, {})
        if not rec.get("has_hit"):
            continue
        tpk = rec.get("hit_t_manual", rec.get("tpk"))
        T = rec.get("T")
        if tpk is not None and T is not None and not (0 <= float(tpk) < float(T)):
            tpk_bad.append(s)
    g_tr = {well_group(s) for s in train}
    g_va = {well_group(s) for s in val}
    return {
        "n_train": len(train),
        "n_val": len(val),
        "stem_overlap_count": len(stem_overlap),
        "stem_overlap": stem_overlap,
        "unlabeled_count": len(missing),
        "unlabeled": missing[:20],
        "tpk_out_of_range_count": len(tpk_bad),
        "tpk_out_of_range": tpk_bad[:20],
        "well_group_overlap_count": len(g_tr & g_va),
        "well_group_overlap": sorted(g_tr & g_va),
    }


def main(argv):
    data_dir = argv[0] if argv else _read_env_field_dir()
    if not data_dir or not Path(data_dir).exists():
        print(f"ERROR: data dir not found: {data_dir}")
        return 2

    labels = _load_labels()
    print(f"data_dir={data_dir}")
    print(f"labels={len(labels)} entries")

    results = validate_dir(data_dir, labels)
    summary = write_reports(results, RESULTS)
    split_audit = audit_split(labels)

    combined = {"validation": summary, "split_audit": split_audit}
    (RESULTS / "experiment_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== VALIDATION ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== SPLIT AUDIT ===")
    print(json.dumps(split_audit, indent=2, ensure_ascii=False))
    print(f"\nreports -> {RESULTS}")

    hard_fail = summary["n_error"] > 0 or split_audit["stem_overlap_count"] > 0 \
        or split_audit["tpk_out_of_range_count"] > 0
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

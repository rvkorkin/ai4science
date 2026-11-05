"""Normalize label time fields to the 1-minute grid of the enriched npz.

Some records (notably CrownQuest) store T / tpk in raw sampling units while the
enriched npz is already decimated to minutes, so label["T"] != strain.shape[0].
This aligns label["T"] to the real minute length and, for the raw-units case
(ratio >= 2), rescales integer time fields by the decimation factor. Spatial
fields (D, band, n_depth, ...) are left untouched. Originals are backed up to
*.bak before writing.

Run:
    python fix_labels.py            # uses FIELD_DATA_DIR from the fork .env
    python fix_labels.py --dry      # report only, do not write
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TIME_FIELDS = ("tpk", "hit_t_manual")
LABEL_FILES = [PROJECT / "deploy" / "field_strongest_hits.json",
               PROJECT / "core" / "field_strongest_hits.json"]


def _field_dir() -> str | None:
    env = PROJECT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("FIELD_DATA_DIR"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _npz_T(data_dir: Path, stem: str) -> int | None:
    p = data_dir / f"{stem}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        if "strain" not in z.files:
            return None
        return int(z["strain"].shape[0])


def plan_fixes(labels: dict, data_dir: Path) -> list[dict]:
    changes = []
    for stem, rec in labels.items():
        if stem.startswith("_") or not isinstance(rec, dict):
            continue
        label_T = rec.get("T")
        if label_T is None:
            continue
        npz_T = _npz_T(data_dir, stem)
        if npz_T is None or int(label_T) == npz_T:
            continue
        factor = int(round(label_T / npz_T))
        scaled = {}
        if factor >= 2:
            for f in TIME_FIELDS:
                v = rec.get(f)
                if isinstance(v, (int, float)):
                    scaled[f] = int(round(v / factor))
        changes.append({"stem": stem, "old_T": int(label_T), "new_T": npz_T,
                         "factor": factor, "scaled": scaled})
    return changes


def apply_fixes(labels: dict, changes: list[dict]) -> dict:
    by_stem = {c["stem"]: c for c in changes}
    for stem, c in by_stem.items():
        rec = labels[stem]
        rec["T"] = c["new_T"]
        for f, v in c["scaled"].items():
            rec[f] = v
    return labels


def main(argv):
    dry = "--dry" in argv
    data_dir = _field_dir()
    if not data_dir or not Path(data_dir).exists():
        print(f"ERROR: data dir not found: {data_dir}")
        return 2
    data_dir = Path(data_dir)

    labels = json.loads(LABEL_FILES[0].read_text(encoding="utf-8"))
    changes = plan_fixes(labels, data_dir)

    print(f"records to normalize: {len(changes)}")
    for c in changes:
        extra = f"  scaled={c['scaled']}" if c["scaled"] else "  (T-only, drift)"
        print(f"  {c['stem']:28s} T {c['old_T']} -> {c['new_T']}  x{c['factor']}{extra}")

    if dry or not changes:
        print("\n(dry run — nothing written)" if dry else "\nnothing to do")
        return 0

    for path in LABEL_FILES:
        if not path.exists():
            continue
        shutil.copy2(path, path.with_suffix(".json.bak"))
        lab = json.loads(path.read_text(encoding="utf-8"))
        lab = apply_fixes(lab, changes)
        path.write_text(json.dumps(lab, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {path}  (backup: {path.name}.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

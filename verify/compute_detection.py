"""Honest per-FILE detection metrics (AUC, FAR, precision, recall) for the
baseline and soft eval CSVs, reported the way deploy/RUNS.md does: aggregate
each recording by max nn_hit_p, score over FILES, and pick the operating point
by a tolerated false-alarm rate.

CAVEAT printed at the end: this is on split.json (train 430 / val 40), which is
NOT leave-pair-out -- RUNS.md numbers are honest pairloo OOF, so they are a
harder setting and not directly comparable.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core"


def per_file(path: Path):
    """stem -> (has_hit, max nn_hit_p over all rows)."""
    y, s = {}, defaultdict(float)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = r["stem"]
            y[stem] = int(float(r["has_hit"]))
            try:
                p = float(r["nn_hit_p"])
            except (TypeError, ValueError):
                p = 0.0
            s[stem] = max(s[stem], p)
    stems = list(y)
    return (np.array([y[k] for k in stems]),
            np.array([s[k] for k in stems]), stems)


def auc_roc(y, s):
    """Rank-based AUC (Mann-Whitney U), ties averaged. No sklearn dependency."""
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return (r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def op_point(y, s, far_max=0.10):
    """Highest-recall threshold with FAR <= far_max (over files)."""
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    best = None
    for thr in sorted(set(s.tolist()) | {0.0, 1.0}, reverse=True):
        pred = s >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        far = fp / nneg if nneg else 0.0
        rec = tp / npos if npos else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if far <= far_max:
            cand = (rec, thr, far, prec, tp, fp)
            if best is None or cand[0] > best[0]:
                best = cand
    return best  # (recall, thr, far, precision, tp, fp)


def report(name, path, far_max=0.10):
    y, s, stems = per_file(path)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    auc = auc_roc(y, s)
    rec, thr, far, prec, tp, fp = op_point(y, s, far_max)
    print(f"\n== {name} ==  (val files: {len(stems)}  hit={npos}  no-hit={nneg})")
    print(f"  AUC (per file)           : {auc:.3f}")
    print(f"  @FAR<= {far_max:.2f}: thr={thr:.3f}  recall={rec:.3f}  "
          f"precision={prec:.3f}  FAR={far:.3f}  (TP={tp} FP={fp})")
    return {"name": name, "n": len(stems), "pos": npos, "neg": nneg,
            "auc": auc, "thr": thr, "recall": rec, "precision": prec, "far": far}


def main():
    report("baseline", CORE / "nn_seq_eval_baseline.csv")
    report("soft", CORE / "nn_seq_eval_soft.csv")
    print("\nCAVEAT: split.json is NOT leave-pair-out (well-group overlap=20, see HW2).")
    print("deploy/RUNS.md numbers (A/B/C) are honest pairloo OOF -> harder, not comparable 1:1.")


if __name__ == "__main__":
    main()

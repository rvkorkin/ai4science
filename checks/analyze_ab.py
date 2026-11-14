"""Fair A/B of baseline vs soft-label models on the TIME task (val set).

The per-epoch val loss is measured against the manual labels, which is a
*biased referee*: the soft model is trained toward wedge-guarded consensus
targets that deliberately diverge from manual where manual is onset-biased.
So here we score each model's predicted arrival against BOTH references:

  * manual tpk          (the official hand label)
  * consensus t_target  (wedge-guarded robust median, soft_labels_full.csv)

Per val hit-file predicted arrival = median of nn_pred_hit_min over the
pre-hit rows (true remaining L_min > 0). We report MAE (min) of the arrival
error against each reference, plus the detection rate (max nn_hit_p >= 0.5
somewhere pre-hit), for baseline and soft.

Run:
    python analyze_ab.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core"


def _read_eval(path: Path):
    """stem -> list of per-timestep dict(float) rows."""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = r["stem"]
            def fl(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            rows[stem].append({
                "has_hit": int(float(r["has_hit"])),
                "tpk": fl("tpk"), "t_now": fl("t_now"), "L_min": fl("L_min"),
                "pred": fl("nn_pred_hit_min"), "p": fl("nn_hit_p"),
            })
    return rows


def _load_consensus():
    out = {}
    with open(HERE / "results" / "soft_labels_full.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["stem"]] = float(r["t_target"])
    return out


NEARFIELD_MIN = 30.0   # score arrival only within this many min of the true hit


def per_file_arrival(rows):
    """Median predicted arrival in the NEAR-FIELD (true L_min in (0, 30]) --
    this is where the ETA has to be right; far-field early rows (where the hit
    is not yet observable) are excluded so the metric is not dominated by them.
    Falls back to all pre-hit rows if no near-field rows exist."""
    pre = [x for x in rows if x["L_min"] is not None and x["L_min"] > 0 and x["pred"] is not None]
    near = [x for x in pre if x["L_min"] <= NEARFIELD_MIN]
    use = near if near else pre
    if not use:
        return None, False
    arrival = float(np.median([x["pred"] for x in use]))
    det = any((x["p"] or 0.0) >= 0.5 for x in pre)
    return arrival, det


def score(eval_path: Path, consensus: dict):
    rows = _read_eval(eval_path)
    e_man, e_con, dets, n = [], [], [], 0
    for stem, rs in rows.items():
        if not rs or rs[0]["has_hit"] != 1:
            continue
        tpk = rs[0]["tpk"]
        if tpk is None:
            continue
        arrival, det = per_file_arrival(rs)
        if arrival is None:
            continue
        n += 1
        e_man.append(abs(arrival - tpk))
        if stem in consensus:
            e_con.append(abs(arrival - consensus[stem]))
        dets.append(1.0 if det else 0.0)
    return {
        "n_hit": n,
        "mae_manual": float(np.mean(e_man)) if e_man else float("nan"),
        "mae_consensus": float(np.mean(e_con)) if e_con else float("nan"),
        "median_manual": float(np.median(e_man)) if e_man else float("nan"),
        "median_consensus": float(np.median(e_con)) if e_con else float("nan"),
        "detect_rate": float(np.mean(dets)) if dets else float("nan"),
    }


def main():
    consensus = _load_consensus()
    base = score(CORE / "nn_seq_eval_baseline.csv", consensus)
    soft = score(CORE / "nn_seq_eval.csv", consensus)  # soft is the latest eval

    print(f"val hit files scored: baseline={base['n_hit']}  soft={soft['n_hit']}")
    print(f"{'metric':24s} {'baseline':>10s} {'soft':>10s}   winner")
    for key, label in [("mae_manual", "MAE arrival vs manual"),
                       ("median_manual", "median vs manual"),
                       ("mae_consensus", "MAE arrival vs consensus"),
                       ("median_consensus", "median vs consensus"),
                       ("detect_rate", "detect rate (p>=0.5)")]:
        b, s = base[key], soft[key]
        if key == "detect_rate":
            win = "soft" if s > b else ("baseline" if b > s else "tie")
        else:
            win = "soft" if s < b else ("baseline" if b < s else "tie")
        print(f"{label:24s} {b:>10.2f} {s:>10.2f}   {win}")

    out = HERE / "results" / "ab_time_eval.json"
    out.write_text(json.dumps({"baseline": base, "soft": soft}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

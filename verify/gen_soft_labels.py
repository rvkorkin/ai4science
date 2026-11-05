"""Generate per-file SOFT LABELS for every hit file in the train/val split.

For each has_hit recording we gather independent, model-free time/depth
estimates and robustly combine them (median + MAD, |z|>2 trimmed):

  time  estimators : manual-label tpk, activity peak, wedge_sr collapse
  depth estimators : manual-label depth, activity depth, wedge_sr apex, wedge_str apex

Design choices (TIME is the primary target):
  * t_target = robust median of {manual, act, wedge_sr}. This honors the
    Leggett wedge-collapse arrival (wedge_sr is in the mix) but is guarded:
    if one estimator is an outlier (e.g. a manual onset-bias like stg28=330
    while act+wedge agree ~144) it is trimmed, so we never blindly trust a
    single noisy source. If estimators agree, t_target == that time.
  * wedge_str TIME is deliberately excluded (empirically the noisiest; it
    grabs spurious shallow wedges). It is kept for DEPTH only.
  * sigma_t -> per-file tpk jitter width; w_time = 1/(1+sigma_t/T_SCALE)
    -> per-file time-loss weight. Same for depth (secondary).

Fallback: if only the manual label is available, t_target = manual, with a
default sigma so the file is still trusted.

Writes results/soft_labels_full.csv (schema matches nn_seq_pipeline loader).

Run:
    python gen_soft_labels.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from consensus_uncertainty import _robust, T_SCALE, D_SCALE
from localize import localize as act_localize
from wedge_localize import localize_wedge

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent / "deploy"
DEFAULT_SIGMA_T = 5.0     # min, used when only 1 time estimator exists
DEFAULT_SIGMA_D = 50.0    # m,   used when only 1 depth estimator exists


def _field_dir() -> Path:
    env = HERE.parent / ".env"
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("FIELD_DATA_DIR"):
            return Path(line.split("=", 1)[1].strip().strip('"').strip("'"))
    raise RuntimeError("FIELD_DATA_DIR not set")


def estimate(stem: str, data: Path, rec: dict):
    p = data / f"{stem}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        strain = z["strain"].astype(np.float64)
        depth = z["depth"].astype(np.float64)
        sr = z["strain_rate"].astype(np.float64) if "strain_rate" in z.files else None
    warm = max(3, strain.shape[0] // 5)

    tnames, tvals, dnames, dvals = [], [], [], []

    tpk = rec.get("hit_t_manual", rec.get("tpk"))
    if tpk is not None:
        tnames.append("manual"); tvals.append(float(tpk))
    hd = rec.get("hit_depth_m")
    if hd is not None:
        dnames.append("manual"); dvals.append(float(hd))

    r = act_localize(p)
    tnames.append("act"); tvals.append(float(r["t_hit"]))
    dnames.append("act"); dvals.append(float(r["depth_hit_m"]))

    if sr is not None:
        rs = localize_wedge(sr, depth, warm)
        if rs is not None:
            tnames.append("wedge_sr"); tvals.append(float(rs["hit_time"]))
            dnames.append("wedge_sr"); dvals.append(float(rs["hit_depth"]))
    rt = localize_wedge(strain, depth, warm)
    if rt is not None:
        dnames.append("wedge_str"); dvals.append(float(rt["hit_depth"]))  # depth only

    return tnames, np.array(tvals, float), dnames, np.array(dvals, float)


def combine(vals: np.ndarray, names: list[str], default_sigma: float):
    if vals.size == 0:
        return None
    if vals.size == 1:
        return float(vals[0]), float(default_sigma)
    center, sigma, _keep = _robust(vals, names)
    if sigma < 1e-6:
        sigma = default_sigma * 0.1          # near-perfect agreement, tiny floor
    return float(center), float(sigma)


def main(argv):
    data = _field_dir()
    labels = json.load(open(DEPLOY / "field_strongest_hits.json", encoding="utf-8"))
    sp = json.load(open(DEPLOY / "split.json", encoding="utf-8"))
    stems = list(dict.fromkeys(list(sp["train"]) + list(sp["val"])))

    rows, n_hit, n_done, n_skip = [], 0, 0, 0
    for stem in stems:
        rec = labels.get(stem, {})
        if not rec.get("has_hit"):
            continue
        n_hit += 1
        est = estimate(stem, data, rec)
        if est is None:
            n_skip += 1
            continue
        tnames, tvals, dnames, dvals = est
        ct = combine(tvals, tnames, DEFAULT_SIGMA_T)
        cd = combine(dvals, dnames, DEFAULT_SIGMA_D)
        if ct is None:
            n_skip += 1
            continue
        t_tgt, sig_t = ct
        d_tgt, sig_d = cd if cd is not None else (float("nan"), DEFAULT_SIGMA_D)
        rows.append({
            "stem": stem, "has_hit": 1,
            "t_target": round(t_tgt, 1), "sigma_t": round(sig_t, 2),
            "w_time": round(1.0 / (1.0 + sig_t / T_SCALE), 3),
            "d_target": round(d_tgt, 1), "sigma_d": round(sig_d, 2),
            "w_depth": round(1.0 / (1.0 + sig_d / D_SCALE), 3),
            "n_t": len(tnames), "n_d": len(dnames),
        })
        n_done += 1
        if n_done % 25 == 0:
            print(f"  ... {n_done} done", flush=True)

    out = HERE / "results" / "soft_labels_full.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stem", "has_hit", "t_target", "sigma_t", "w_time",
            "d_target", "sigma_d", "w_depth", "n_t", "n_d"])
        w.writeheader()
        w.writerows(rows)

    wt = np.array([r["w_time"] for r in rows]) if rows else np.array([0.0])
    print(f"\nhit files in split: {n_hit}  |  soft-labelled: {n_done}  |  skipped: {n_skip}")
    print(f"w_time: min={wt.min():.2f} median={np.median(wt):.2f} max={wt.max():.2f}  "
          f"(<0.5: {(wt < 0.5).sum()} files down-weighted hard)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

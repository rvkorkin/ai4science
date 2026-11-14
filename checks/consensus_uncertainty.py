"""Per-recording label uncertainty from estimator disagreement.

Idea (addresses "labels have uncertainty"): a frac-hit label (t, depth) is not
a point, it is a noisy estimate. We get several *independent, model-free*
estimates of (hit_time, hit_depth) and treat their spread as the measured
label noise sigma. That sigma is exactly what feeds a soft/probabilistic label
and a Gaussian-NLL head (per-sample log_sigma) or a per-sample loss weight.

Estimators (all causal, none use the trained/leaked model):
  human  -- hand-eyeballed guess (localize.GUESS)
  act    -- differential-activity peak + activity depth profile (localize.localize)
  wedge_sr  -- converging-FWHM collapse on strain_rate  (Leggett: hit = w->0)
  wedge_str -- converging-FWHM collapse on strain

For each recording we report:
  t_med, d_med        -- consensus (median across estimators)
  sigma_t, sigma_d    -- spread (std, population) == label noise
  spread_t, spread_d  -- max-min range
  w_sample            -- suggested loss weight in (0,1], smaller == noisier
                         label == trust it less (w = 1/(1 + sigma_d/D_SCALE)).

Writes results/consensus_uncertainty.csv and prints a table.

Run:
    python consensus_uncertainty.py STEM [STEM ...]
    python consensus_uncertainty.py            # all GUESS hit-files
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from localize import GUESS, NO_HIT, _field_dir, localize
from wedge_localize import localize_wedge

# PRIMARY target is TIME, depth is secondary.
# time scale (min) mapping sigma_time -> a ~0.5 loss weight for the time head
T_SCALE = 20.0
# depth scale (m) for the secondary depth head weight
D_SCALE = 100.0


def _load(path: Path):
    with np.load(path, allow_pickle=True) as z:
        strain = z["strain"].astype(np.float64)
        depth = z["depth"].astype(np.float64)
        sr = z["strain_rate"].astype(np.float64) if "strain_rate" in z.files else None
    return strain, depth, sr


def estimators(stem: str, data: Path) -> dict:
    """All available (name -> (t, depth)) estimates for one recording."""
    p = data / f"{stem}.npz"
    strain, depth, sr = _load(p)
    warm = max(3, strain.shape[0] // 5)
    out: dict[str, tuple[float, float]] = {}

    g = GUESS.get(stem)
    if g is not None:
        out["human"] = (float(g[0]), float(g[1]))

    r = localize(p)
    out["act"] = (float(r["t_hit"]), float(r["depth_hit_m"]))

    if sr is not None:
        rs = localize_wedge(sr, depth, warm)
        if rs is not None:
            out["wedge_sr"] = (float(rs["hit_time"]), float(rs["hit_depth"]))
    rt = localize_wedge(strain, depth, warm)
    if rt is not None:
        out["wedge_str"] = (float(rt["hit_time"]), float(rt["hit_depth"]))
    return out


MAD_K = 1.4826   # MAD -> sigma (gaussian consistency)
Z_OUT = 2.0      # |x - median| / (MAD*K) above this == outlier estimator


def _robust(vals: np.ndarray, names: list[str]):
    """Median center + MAD sigma with |z|>Z_OUT estimators trimmed.

    Returns (center, sigma, kept_mask). sigma is MAD-based on the kept
    points (falls back to std if the kept MAD is 0, e.g. only 2 left).
    """
    v = np.asarray(vals, dtype=float)
    med = np.median(v)
    scale = MAD_K * np.median(np.abs(v - med))
    if scale <= 1e-9:                       # >= half the estimators agree exactly
        keep = np.ones(v.size, bool)
    else:
        keep = np.abs(v - med) <= Z_OUT * scale
        if keep.sum() < 2:                  # never trim below 2 points
            keep = np.ones(v.size, bool)
    vk = v[keep]
    cen = float(np.median(vk))
    mad_k = MAD_K * float(np.median(np.abs(vk - cen)))
    sig = mad_k if mad_k > 1e-9 else float(np.std(vk))
    return cen, sig, keep


def consensus(est: dict) -> dict:
    names = list(est)
    ts = np.array([est[k][0] for k in names], dtype=float)
    ds = np.array([est[k][1] for k in names], dtype=float)
    t_med, sig_t, keep_t = _robust(ts, names)
    d_med, sig_d, keep_d = _robust(ds, names)
    drop_t = [names[i] for i in range(len(names)) if not keep_t[i]]
    drop_d = [names[i] for i in range(len(names)) if not keep_d[i]]
    # physics time anchor (Leggett wedge collapse on strain_rate), and how far
    # the other estimators sit from it -- the honest "time arrival" spread.
    t_anchor = float(est["wedge_sr"][0]) if "wedge_sr" in est else float("nan")
    return {
        "n_est": int(ts.size),
        "t_med": t_med,
        "d_med": d_med,
        "sigma_t": float(sig_t),
        "sigma_d": float(sig_d),
        "spread_t": float(ts.max() - ts.min()),
        "spread_d": float(ds.max() - ds.min()),
        "t_anchor_sr": t_anchor,
        "w_time": float(1.0 / (1.0 + sig_t / T_SCALE)),    # PRIMARY
        "w_depth": float(1.0 / (1.0 + sig_d / D_SCALE)),   # secondary
        "dropped_t": "|".join(drop_t),
        "dropped_d": "|".join(drop_d),
    }


def main(argv):
    data = _field_dir()
    stems = argv if argv else [s for s in GUESS if s not in NO_HIT]

    computed = []
    for stem in stems:
        p = data / f"{stem}.npz"
        if not p.exists():
            print(f"{stem:24s}  (npz not found)")
            continue
        est = estimators(stem, data)
        computed.append((stem, est, consensus(est)))

    # PRIMARY = time: list the time-noisiest (lowest w_time) first
    computed.sort(key=lambda t: t[2]["w_time"])

    rows = []
    print(f"{'recording':24s} | {'t_med':>6s} {'t_sr':>5s} {'sig_t':>6s} {'w_t':>5s}"
          f" || {'d_med':>7s} {'sig_d':>6s} {'w_d':>5s}   estimators(t,d)")
    for stem, est, c in computed:
        dt, dd = set(c["dropped_t"].split("|")), set(c["dropped_d"].split("|"))

        def _mark(k):
            m = ("t" if k in dt else "") + ("d" if k in dd else "")
            return f"{k}{'[drop:' + m + ']' if m else ''}"

        detail = "  ".join(f"{_mark(k)}={v[0]:.0f}/{v[1]:.0f}" for k, v in est.items())
        print(f"{stem:24s} | {c['t_med']:>6.0f} {c['t_anchor_sr']:>5.0f} {c['sigma_t']:>6.1f}"
              f" {c['w_time']:>5.2f} || {c['d_med']:>7.0f} {c['sigma_d']:>6.0f}"
              f" {c['w_depth']:>5.2f}   {detail}")
        row = {"recording": stem, **c}
        for k, v in est.items():
            row[f"{k}_t"] = v[0]
            row[f"{k}_d"] = v[1]
        rows.append(row)

    if rows:
        out = Path(__file__).resolve().parent / "results" / "consensus_uncertainty.csv"
        out.parent.mkdir(exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        head = ["recording", "n_est", "t_med", "t_anchor_sr", "sigma_t", "spread_t",
                "w_time", "d_med", "sigma_d", "spread_d", "w_depth"]
        head += [k for k in keys if k not in head]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=head)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")

        # ---- training-ready soft labels (TIME is the primary target) --------
        # t_target = wedge-collapse time (Leggett arrival), NOT the median
        # (median is biased early by onset-type estimators). sigma_t -> per-file
        # jitter width, w_time/w_depth -> per-file loss weights.
        soft = out.parent / "soft_labels.csv"
        with soft.open("w", newline="", encoding="utf-8") as f:
            sw = csv.DictWriter(f, fieldnames=[
                "stem", "has_hit", "t_target", "sigma_t", "w_time",
                "d_target", "sigma_d", "w_depth"])
            sw.writeheader()
            for r in rows:
                t_tgt = r.get("t_anchor_sr")
                if t_tgt is None or not np.isfinite(t_tgt):
                    t_tgt = r["t_med"]          # fallback if wedge did not converge
                sw.writerow({
                    "stem": r["recording"], "has_hit": 1,
                    "t_target": round(float(t_tgt), 1),
                    "sigma_t": round(float(r["sigma_t"]), 2),
                    "w_time": round(float(r["w_time"]), 3),
                    "d_target": round(float(r["d_med"]), 1),
                    "sigma_d": round(float(r["sigma_d"]), 2),
                    "w_depth": round(float(r["w_depth"]), 3),
                })
        print(f"wrote {soft}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

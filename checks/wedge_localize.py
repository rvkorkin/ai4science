"""Wedge-based (converging-FWHM) localization of a frac-hit. Model-free.

Physics: a real hit is a wedge that converges in depth-time. We track the
FWHM core width w(t) along depth (deploy/wedge_near.width_series), the apex
channel (argmax z) = hit depth, and extrapolate w -> 0 (wedge_near.eta_series)
to get the collapse TIME. Timing comes from convergence, not from peak
amplitude. Reported for strain_rate (active stretch, ZSRL) and strain.

Run:
    python wedge_localize.py STEM [STEM ...]
    python wedge_localize.py           # built-in candidate list
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# wedge_near.py (FWHM-collapse physics core) ships in src/ of this repo.
DEPLOY = HERE.parent / "src"
sys.path.insert(0, str(DEPLOY))
import wedge_near as wn  # noqa: E402

from localize import GUESS, EXTRA, _field_dir  # noqa: E402


def wedge_track(field: np.ndarray, depth: np.ndarray, warm: int):
    a = wn._gaussian_filter1d(np.nan_to_num(field).astype(np.float64), wn.DEPTH_SIGMA, axis=1)
    w0 = a[:max(3, warm)]
    med = np.median(w0, axis=0)
    sc = np.median(np.abs(w0 - med), axis=0) * 1.4826 + 1e-6
    z = np.abs((a - med) / sc)
    T = z.shape[0]
    w = np.full(T, np.nan)
    apex = np.full(T, np.nan)
    for t in range(T):
        row = z[t]
        pk = float(row.max())
        if pk < wn.Z_THR:
            continue
        j = int(row.argmax())
        apex[t] = depth[j]
        half = 0.5 * pk
        lo = j
        while lo > 0 and row[lo - 1] >= half:
            lo -= 1
        hi = j
        while hi < len(row) - 1 and row[hi + 1] >= half:
            hi += 1
        w[t] = abs(float(depth[hi] - depth[lo]))
    fill = np.nanmedian(w) if np.isfinite(w).any() else 0.0
    w_s = wn._uniform_filter1d(np.nan_to_num(w, nan=fill), wn.SMOOTH)
    eta = wn.eta_series(w_s)
    return w_s, apex, eta


def localize_wedge(field, depth, warm):
    w_s, apex, eta = wedge_track(field, depth, warm)
    conv = np.isfinite(eta) & (eta > 0)
    if not conv.any():
        return None
    idx = np.where(conv)[0]
    t_c = int(idx[np.argmin(eta[idx])])          # closest approach
    hit_time = min(len(w_s) - 1, t_c + float(eta[t_c]))
    lo = max(0, t_c - 10)
    ap = apex[lo:t_c + 1]
    ap = ap[np.isfinite(ap)]
    hit_depth = float(np.median(ap)) if ap.size else float(apex[t_c]) if np.isfinite(apex[t_c]) else np.nan
    return {"t_c": t_c, "eta": round(float(eta[t_c]), 1),
            "hit_time": round(float(hit_time), 1),
            "hit_depth": round(hit_depth, 1), "min_w": round(float(w_s[t_c]), 1)}


def main(argv):
    data = _field_dir()
    stems = argv if argv else list(GUESS) + EXTRA
    hdr = (f"{'recording':24s} {'T':>4s} | {'SR: t_hit':>9s} {'depth':>7s} {'wmin':>5s}"
           f" | {'STR: t_hit':>10s} {'depth':>7s}  guess(t,depth)")
    print(hdr)
    for stem in stems:
        p = data / f"{stem}.npz"
        if not p.exists():
            print(f"{stem:24s}  (npz not found)")
            continue
        with np.load(p, allow_pickle=True) as z:
            strain = z["strain"].astype(np.float64)
            depth = z["depth"].astype(np.float64)
            sr = z["strain_rate"].astype(np.float64) if "strain_rate" in z.files else None
        warm = max(3, strain.shape[0] // 5)
        rsr = localize_wedge(sr, depth, warm) if sr is not None else None
        rst = localize_wedge(strain, depth, warm)
        g = GUESS.get(stem)
        gs = f"{g[0]},{g[1]}" if g else "-"

        def fmt(r):
            return (f"{r['hit_time']:>7}m {r['hit_depth']:>7.1f} {r['min_w']:>5.1f}"
                    if r else f"{'no conv':>7} {'-':>7} {'-':>5}")

        def fmt2(r):
            return (f"{r['hit_time']:>8}m {r['hit_depth']:>7.1f}"
                    if r else f"{'no conv':>8} {'-':>7}")

        print(f"{stem:24s} {strain.shape[0]:>4d} | {fmt(rsr)} | {fmt2(rst)}  {gs}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

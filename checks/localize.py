"""Model-free causal localization of a putative frac-hit in a record.

Deliberately does NOT use the trained (leaked) model. Time = causal peak of
depth-aggregated differential activity; depth = differential-activity profile
around that time (same idea as core.windowed_core.hit_depth_auto). Prints the
estimate next to any hand-eyeballed guess.

Run:
    python localize.py STEM [STEM ...]
    python localize.py            # built-in candidate list
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

W_ACT = 6

GUESS = {
    "VYD_Trt10_1WA-2WA": (106, 6055),
    "forge7": (52, 2875),
    "HFTS2_B1H_B3H_stg17": (220, 5140),
    "HFTS2_B3H_B4H_stg24": (180, 4990),
    "HFTS2_B3H_B4H_stg28": (330, 4266),
}
# recordings the human review concluded have NO frac-hit (no converging wedge)
NO_HIT = {"forge7"}
EXTRA = ["VYD_Trt9_1WA-2WA", "HFTS2_T16H_B4H_stg25", "VYD_Trt21_1WA-2WA",
         "VYD_Trt16_1WA-2WA", "HFTS2_T16H_B4H_stg09"]


def _field_dir() -> Path:
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("FIELD_DATA_DIR"):
            return Path(line.split("=", 1)[1].strip().strip('"').strip("'"))
    raise RuntimeError("FIELD_DATA_DIR not set")


def activity_time_series(strain: np.ndarray) -> np.ndarray:
    """Causal p95-over-channels of |strain[t]-strain[t-W]|, in MAD over baseline."""
    T = strain.shape[0]
    act = np.full(T, np.nan)
    if T > W_ACT:
        ch = strain[W_ACT:] - strain[:-W_ACT]
        act[W_ACT:] = np.nanpercentile(np.abs(ch), 95, axis=1)
    warm = max(W_ACT + 3, T // 5)
    base_slice = act[W_ACT:warm]
    base_slice = base_slice[np.isfinite(base_slice)]
    base = np.percentile(base_slice, 20) if base_slice.size else 0.0
    mad = (np.median(np.abs(act[np.isfinite(act)] - np.nanmedian(act))) * 1.4826 + 1e-9)
    amp = (act - base) / mad
    return amp


def hit_depth_at(strain: np.ndarray, depth: np.ndarray, tpk: int,
                 pre_win: int = 20, post_win: int = 8) -> float:
    T, D = strain.shape
    ti = int(np.clip(round(tpk), 0, T - 1))
    if T <= W_ACT + 2:
        return float(depth[D // 2])
    ch = np.abs(strain[W_ACT:] - strain[:-W_ACT])
    tt = np.arange(W_ACT, T)
    pre = (tt >= ti - pre_win) & (tt < ti)
    post = (tt >= ti) & (tt < ti + post_win)
    if post.sum() < 2:
        post = (tt >= ti - 2) & (tt < ti + 4)
    if pre.sum() < 2:
        pre = tt < max(W_ACT + 3, ti - 2)
    a_post = np.nanmedian(ch[post], axis=0)
    a_pre = np.nanmedian(ch[pre], axis=0) if pre.any() else np.zeros(D)
    prof = a_post - a_pre
    k = max(1, int(round(7.0 / max(np.median(np.abs(np.diff(depth))), 1e-6))))
    if k >= 2:
        prof = np.convolve(np.nan_to_num(prof), np.ones(k) / k, mode="same")
    prof = prof * (a_post >= np.nanmedian(a_post))
    edge = int(D * 0.05)
    prof[:edge] = -np.inf
    prof[D - edge:] = -np.inf
    return float(depth[int(np.argmax(prof))])


def localize(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        strain = z["strain"].astype(np.float64)
        depth = z["depth"].astype(np.float64)
        pred_depth = float(z["pred_hit_depth_m"]) if "pred_hit_depth_m" in z.files else None
    amp = activity_time_series(strain)
    warm = max(W_ACT + 3, strain.shape[0] // 5)
    sm = np.copy(amp)
    for i in range(len(sm)):
        lo = max(0, i - 2)
        sm[i] = np.nanmedian(amp[lo:i + 1])
    masked = np.where(np.arange(len(sm)) >= warm, sm, -np.inf)
    t_hit = int(np.nanargmax(masked))
    peak = float(sm[t_hit])
    d_hit = hit_depth_at(strain, depth, t_hit)
    return {"T": strain.shape[0], "t_hit": t_hit, "amp_peak": round(peak, 2),
            "depth_hit_m": round(d_hit, 1),
            "npz_pred_depth_m": round(pred_depth, 1) if pred_depth is not None else None}


def main(argv):
    data = _field_dir()
    stems = argv if argv else list(GUESS) + EXTRA
    print(f"{'recording':24s} {'T':>4s} {'t_est':>6s} {'depth_est':>10s} "
          f"{'npz_pred_d':>10s} {'amp':>5s}   guess(t,depth)")
    for stem in stems:
        p = data / f"{stem}.npz"
        if not p.exists():
            print(f"{stem:24s}  (npz not found)")
            continue
        r = localize(p)
        g = GUESS.get(stem)
        gs = f"{g[0]},{g[1]}" if g else "-"
        print(f"{stem:24s} {r['T']:>4d} {r['t_hit']:>6d} {r['depth_hit_m']:>10.1f} "
              f"{str(r['npz_pred_depth_m']):>10s} {r['amp_peak']:>5.1f}   {gs}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

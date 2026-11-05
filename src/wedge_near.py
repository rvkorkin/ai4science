# -*- coding: utf-8 -*-
"""Classic FWHM-collapse wedge ETA -- near-zone physics, NO training.

Self-contained port of NeuralNet/wedge.py, made geometry-independent (runs on
the SAME fixed-size depth windows as the windowed network). The learned model
raises the alarm at a low false-alarm rate; once it is yellow/red, this pure-
physics ETA sharpens the last-minutes time-to-hit.

Observable: the width w(t) of the deformation core along depth. As the fracture
approaches, the wedge converges and w falls; a causal linear extrapolation
w -> 0 gives ETA(t) = w(t) / (-dw/dt) when dw/dt < 0. Computed on both strain
and strain-rate, in every window, and aggregated as the NEAREST converging
wedge (smallest finite ETA) across windows and channels.

The measured behaviour (windowed sweep at FAR=0.10): as a standalone detector
the wedge is weak (it saturates), but as a near-field ETA refiner it is strong
-- so it is used ONLY to shorten the reported ETA when the network is already
alarmed. It never raises the alarm on its own.
"""
from __future__ import annotations

import numpy as np

WARM = 30          # min of background used to build the MAD baseline
Z_THR = 3.0        # z below this is not counted as part of the wedge core
FIT_WIN = 25       # window of the linear regression of w(t), min
MIN_SLOPE = 0.20   # m/min; slower -> treated as "no convergence"
SMOOTH = 7         # smoothing of w(t), min
DEPTH_SIGMA = 2.0  # along-depth gaussian smoothing (channels)


def width_series(a2d: np.ndarray, depth: np.ndarray, warm: int):
    """w(t) = FWHM of the deformation core along depth (NaN if none)."""
    # force a clean, contiguous, finite float64 array: numpy's median/partition
    # (introselect) can fault on non-contiguous / mixed-dtype / NaN-laden views,
    # and that is exactly the op the deploy crash logs point at.
    a2d = np.ascontiguousarray(np.nan_to_num(a2d, copy=False), dtype=np.float64)
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    w0 = a2d[:max(3, warm)]
    med = np.median(w0, axis=0)
    sc = np.median(np.abs(w0 - med), axis=0) * 1.4826 + 1e-6
    z = np.abs((a2d - med) / sc)
    T = z.shape[0]
    w = np.full(T, np.nan)
    for t in range(T):
        row = z[t]
        pk = float(row.max())
        if pk < Z_THR:
            continue
        j = int(row.argmax())
        half = 0.5 * pk
        lo = j
        while lo > 0 and row[lo - 1] >= half:
            lo -= 1
        hi = j
        while hi < len(row) - 1 and row[hi + 1] >= half:
            hi += 1
        w[t] = abs(float(depth[hi] - depth[lo]))
    return w


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y vs x, closed form (no LAPACK call).

    np.polyfit(x, y, 1) -- a full SVD/QR solve via numpy.linalg.lstsq for
    a plain 2-parameter line fit -- is what crashed with an access
    violation deep in LAPACK (deploy/logs/*), called here thousands of
    times a session since eta_series below recomputes the whole history
    every single minute. This is the textbook closed-form OLS slope
    (cov(x,y)/var(x)); mathematically identical result for a degree-1
    fit, but pure numpy arithmetic -- no LAPACK, nothing to crash there.
    """
    xm = x.mean()
    dx = x - xm
    denom = float(np.dot(dx, dx))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    ym = y.mean()
    b = float(np.dot(dx, y - ym)) / denom
    return b if np.isfinite(b) else float("nan")


def eta_series(w: np.ndarray, win: int = FIT_WIN, min_slope: float = MIN_SLOPE):
    """Causal linear extrapolation w -> 0. NaN where no convergence is visible."""
    T = len(w)
    eta = np.full(T, np.nan)
    for t in range(win, T):
        y = w[t - win + 1:t + 1]
        ok = np.isfinite(y)
        if ok.sum() < win * 0.6:
            continue
        x = np.arange(win)[ok]
        b = _slope(x, y[ok])
        if np.isfinite(b) and b < -min_slope and np.isfinite(w[t]) and w[t] > 0:
            eta[t] = w[t] / (-b)
    return eta


def _correlate1d(a: np.ndarray, kernel: np.ndarray, axis: int, pad_mode: str) -> np.ndarray:
    """1D correlate along `axis` with a small fixed-size kernel -- pure numpy.

    scipy.ndimage.gaussian_filter1d / uniform_filter1d call into compiled
    C code that has crashed with a native access violation on this machine
    (deploy/logs/shell_20260911_161320.log, inside
    scipy/ndimage/_filters.py:_gaussian_kernel1d, reached from here). Same
    class of fix as _slope below: these are small, fixed-size linear
    filters, done here via a sliding window + tensordot -- pure numpy
    arithmetic, nothing compiled to crash.
    """
    radius = kernel.size // 2
    a = np.moveaxis(a, axis, -1)
    pad = [(0, 0)] * (a.ndim - 1) + [(radius, radius)]
    padded = np.pad(a, pad, mode=pad_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=-1)
    out = np.tensordot(windows, kernel, axes=([-1], [0]))
    return np.moveaxis(out, -1, axis)


def _gaussian_filter1d(a: np.ndarray, sigma: float, axis: int, truncate: float = 4.0) -> np.ndarray:
    """Drop-in for scipy.ndimage.gaussian_filter1d(..., mode="nearest")."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return _correlate1d(a, kernel, axis, pad_mode="edge")


def _uniform_filter1d(a: np.ndarray, size: int) -> np.ndarray:
    """Drop-in for scipy.ndimage.uniform_filter1d(a, size) (default mode="reflect",
    i.e. edge-duplicating symmetric padding -- numpy's "symmetric" mode)."""
    kernel = np.full(size, 1.0 / size)
    return _correlate1d(a, kernel, axis=-1, pad_mode="symmetric")


def _band_eta(arr: np.ndarray, depth: np.ndarray, warm: int) -> np.ndarray:
    a = _gaussian_filter1d(arr, DEPTH_SIGMA, axis=1)
    w = width_series(a, depth, warm)
    fill = np.nanmedian(w) if np.isfinite(w).any() else 0.0
    w = _uniform_filter1d(np.nan_to_num(w, nan=fill), SMOOTH)
    return eta_series(w)


def eta_over_windows(strain: np.ndarray, strain_rate: np.ndarray,
                     depth: np.ndarray, dt_min: float,
                     windows: list[tuple[int, int, float]]) -> np.ndarray:
    """Aggregate wedge ETA (min) per timestep across windows x {strain, rate}.

    Returns an array of length T; +inf where nothing converges.
    """
    T = strain.shape[0]
    warm = max(3, int(WARM / max(dt_min, 1e-6)))
    agg = np.full(T, np.inf)
    for (i0, i1, _c) in windows:
        if i1 - i0 < 4:
            continue
        for arr in (strain[:, i0:i1], strain_rate[:, i0:i1]):
            e = _band_eta(arr, depth[i0:i1], warm)
            fin = np.isfinite(e)
            if fin.any():
                agg[fin] = np.minimum(agg[fin], e[fin])
    return agg

"""Детерминированные синтетические .npz-фикстуры для автономного сценария.

Фиксированный seed → результат сценария стабилен и воспроизводится под git-меткой.
Сами .npz не коммитятся (gitignore), а генерируются этим скриптом при запуске.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parent / "fixtures"


def _wedge(T=120, D=40, hit_t=90, seed=1):
    """Синтетический сходящийся клин: ширина FWHM стремится к 0 к hit_t."""
    rng = np.random.default_rng(seed)
    depth = np.arange(D) * 2.0
    strain = rng.standard_normal((T, D)) * 1e-3
    sr = rng.standard_normal((T, D)) * 1e-3
    apex = D // 2
    for t in range(T):
        w = max(1.0, (hit_t - t) * 0.4) if t <= hit_t else 1.0
        amp = 1.0 / (1.0 + abs(hit_t - t) * 0.1)
        prof = amp * np.exp(-0.5 * ((np.arange(D) - apex) / w) ** 2)
        sr[t] += prof
        strain[t] += np.cumsum(prof) * 1e-2
    return strain.astype(np.float32), sr.astype(np.float32), depth.astype(np.float64)


def _save(name, strain, sr, depth, dt_sec=60.0):
    FIX.mkdir(parents=True, exist_ok=True)
    np.savez(FIX / name, strain=strain, strain_rate=sr, depth=depth,
             dt_sec=np.float64(dt_sec))


def build():
    # 1. валидный hit (клин сходится ~t=90)
    s, r, d = _wedge(hit_t=90, seed=1)
    _save("rec_hit.npz", s, r, d)

    # 2. валидный no-hit (нет схождения)
    rng = np.random.default_rng(2)
    T, D = 120, 40
    s2 = (rng.standard_normal((T, D)) * 1e-3).astype(np.float32)
    r2 = (rng.standard_normal((T, D)) * 1e-3).astype(np.float32)
    d2 = (np.arange(D) * 2.0).astype(np.float64)
    _save("rec_nohit.npz", s2, r2, d2)

    # 3. битый: отсутствует strain_rate
    FIX.mkdir(parents=True, exist_ok=True)
    np.savez(FIX / "bad_missing_key.npz", strain=s2, depth=d2, dt_sec=np.float64(60.0))

    # 4. битый: NaN в strain
    s3 = s2.copy()
    s3[0, 0] = np.nan
    _save("bad_nan.npz", s3, r2, d2)

    # 5. битый: dt_sec <= 0
    _save("bad_dt.npz", s2, r2, d2, dt_sec=0.0)

    return sorted(p.name for p in FIX.glob("*.npz"))


if __name__ == "__main__":
    print("\n".join(build()))

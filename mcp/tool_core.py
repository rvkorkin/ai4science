"""Чистая логика инструмента frac_hit_verify (без побочных эффектов).

Загружает запись LF-DAS (.npz), прогоняет валидатор данных из ДЗ-2 и, если данные
валидны, оценивает время прихода frac-hit по кинематике клина (FWHM-collapse).
Возвращает JSON-сериализуемый вердикт. Детерминирован.

Автор: агент (Cursor), переиспользует verify/validate_data.py и src/wedge_near.py
(человек: постановка, физика клина, ревью).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "verify", _ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from validate_data import validate_npz          # noqa: E402
from wedge_near import eta_over_windows          # noqa: E402


def frac_hit_verify(npz_path: str) -> dict:
    """Проверить запись LF-DAS и оценить приход frac-hit. Возвращает вердикт-dict."""
    v = validate_npz(npz_path)
    out = {
        "file": Path(npz_path).name,
        "valid": bool(v["ok"]),
        "errors": list(v["errors"]),
        "warnings": list(v["warnings"]),
        "stats": dict(v["stats"]),
        "hit": {"estimated": False},
    }
    if not v["ok"]:
        return out

    try:
        z = np.load(npz_path, allow_pickle=False)
        strain = z["strain"].astype(float)
        sr = z["strain_rate"].astype(float)
        depth = z["depth"].astype(float)
        dt_sec = float(np.ravel(z["dt_sec"])[0])
        dt_min = dt_sec / 60.0
        D = strain.shape[1]
        windows = [(0, D, 0.0)]  # единое окно на всю глубину
        eta = eta_over_windows(strain, sr, depth, dt_min, windows)
        t_idx = np.arange(len(eta), dtype=float)
        finite = np.isfinite(eta)
        if finite.any():
            # прогноз абсолютного времени прихода (мин) на каждом таймстепе = t + eta(t)
            pred = t_idx[finite] * dt_min + eta[finite]
            tc = int(t_idx[finite][int(np.argmin(pred))])  # таймстеп ближайшего подхода
            out["hit"] = {
                "estimated": True,
                "hit_time_min": round(float(np.median(pred)), 2),
                "closest_approach_idx": tc,
                "n_converging_steps": int(finite.sum()),
            }
    except Exception as e:  # noqa: BLE001
        out["warnings"].append(f"wedge_estimate_failed:{e}")
    return out

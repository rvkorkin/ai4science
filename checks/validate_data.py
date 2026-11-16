"""Data validator for enriched DAS records (field_data_enriched/*.npz).

Checks each record against the schema and physical constraints the pipeline
assumes, and cross-checks the label file. Importable (validate_npz,
validate_dir) and runnable as a CLI.

Usage:
    python validate_data.py <data_dir> [labels.json] [--out results]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("strain", "depth", "strain_rate", "dt_sec")
DZ_MIN_M, DZ_MAX_M = 0.3, 6.0
DT_EXPECTED_S = 60.0
SR_ABS_WARN = 1.0e4


def _scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = z[key]
    try:
        return float(np.asarray(v).reshape(-1)[0])
    except Exception:
        return default


def validate_npz(path: str | Path, labels: dict | None = None) -> dict:
    path = Path(path)
    stem = path.stem
    res = {"stem": stem, "ok": True, "errors": [], "warnings": [], "stats": {}}

    def err(m):
        res["errors"].append(m)
        res["ok"] = False

    def warn(m):
        res["warnings"].append(m)

    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        err(f"load_failed: {e}")
        return res

    for k in REQUIRED_KEYS:
        if k not in z.files:
            err(f"missing_key:{k}")
    if not res["ok"]:
        return res

    strain = z["strain"]
    sr = z["strain_rate"]
    depth = z["depth"]
    dt = _scalar(z, "dt_sec")

    if strain.ndim != 2:
        err(f"strain_ndim:{strain.ndim}")
        return res
    T, D = strain.shape
    res["stats"].update(T=int(T), D=int(D))

    if sr.shape != strain.shape:
        err(f"strain_rate_shape:{sr.shape}!=strain{strain.shape}")
    if depth.ndim != 1 or depth.shape[0] != D:
        err(f"depth_shape:{depth.shape}!=({D},)")

    n_nan_s = int(np.count_nonzero(~np.isfinite(strain)))
    n_nan_sr = int(np.count_nonzero(~np.isfinite(sr)))
    res["stats"].update(nan_strain=n_nan_s, nan_strain_rate=n_nan_sr)
    if n_nan_s:
        err(f"nonfinite_strain:{n_nan_s}")
    if n_nan_sr:
        err(f"nonfinite_strain_rate:{n_nan_sr}")

    if dt is None or not np.isfinite(dt) or dt <= 0:
        err(f"dt_sec_invalid:{dt}")
    else:
        res["stats"]["dt_sec"] = dt
        if abs(dt - DT_EXPECTED_S) > 1e-6:
            warn(f"dt_sec_unexpected:{dt}")

    if depth.ndim == 1 and depth.shape[0] == D and D >= 2:
        dd = np.diff(depth.astype(float))
        if not (np.all(dd > 0) or np.all(dd < 0)):
            warn("depth_not_monotonic")
        dz = float(np.median(np.abs(dd)))
        res["stats"]["dz_m"] = dz
        if not (DZ_MIN_M <= dz <= DZ_MAX_M):
            warn(f"dz_out_of_range:{dz:.3f}")

    if "dist_profile_m" in z.files:
        dp = z["dist_profile_m"]
        if dp.shape != (D,):
            err(f"dist_profile_shape:{dp.shape}!=({D},)")

    if n_nan_sr == 0 and sr.size:
        sr_abs_max = float(np.max(np.abs(sr)))
        res["stats"]["sr_abs_max"] = sr_abs_max
        if sr_abs_max > SR_ABS_WARN:
            warn(f"strain_rate_extreme:{sr_abs_max:.1f}")

    if labels is not None:
        rec = labels.get(stem)
        if rec is None:
            warn("no_label_entry")
        else:
            has_hit = bool(rec.get("has_hit"))
            res["stats"]["has_hit"] = has_hit
            lt, ld = rec.get("T"), rec.get("D")
            if lt is not None and int(lt) != int(T):
                warn(f"label_T_mismatch:{lt}!={T}")
            if ld is not None and int(ld) != int(D):
                warn(f"label_D_mismatch:{ld}!={D}")
            if has_hit:
                tpk = rec.get("hit_t_manual", rec.get("tpk"))
                if tpk is None:
                    err("hit_without_tpk")
                elif not (0 <= float(tpk) < float(T)):
                    err(f"tpk_out_of_range:{tpk}/{T}")

    return res


def validate_dir(data_dir: str | Path, labels: dict | None = None) -> list[dict]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))
    return [validate_npz(p, labels) for p in files]


def summarize(results: list[dict]) -> dict:
    from collections import Counter

    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    err_types = Counter()
    warn_types = Counter()
    for r in results:
        for e in r["errors"]:
            err_types[e.split(":")[0]] += 1
        for w in r["warnings"]:
            warn_types[w.split(":")[0]] += 1
    hits = sum(1 for r in results if r["stats"].get("has_hit") is True)
    nohits = sum(1 for r in results if r["stats"].get("has_hit") is False)
    Ts = [r["stats"].get("T") for r in results if r["stats"].get("T")]
    Ds = [r["stats"].get("D") for r in results if r["stats"].get("D")]
    dzs = [r["stats"].get("dz_m") for r in results if r["stats"].get("dz_m")]
    return {
        "n_files": n,
        "n_ok": n_ok,
        "n_error": n - n_ok,
        "error_types": dict(err_types),
        "warning_types": dict(warn_types),
        "n_hit": hits,
        "n_nohit": nohits,
        "T_min": min(Ts) if Ts else None,
        "T_max": max(Ts) if Ts else None,
        "D_values": sorted(set(Ds)),
        "dz_median_m": round(float(np.median(dzs)), 3) if dzs else None,
    }


def write_reports(results: list[dict], out_dir: str | Path) -> dict:
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (out_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cols = ["stem", "ok", "T", "D", "dt_sec", "dz_m", "has_hit",
            "nan_strain", "nan_strain_rate", "sr_abs_max", "errors", "warnings"]
    with open(out_dir / "validation_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            s = r["stats"]
            w.writerow([
                r["stem"], int(r["ok"]), s.get("T"), s.get("D"), s.get("dt_sec"),
                s.get("dz_m"), s.get("has_hit"), s.get("nan_strain"),
                s.get("nan_strain_rate"), s.get("sr_abs_max"),
                ";".join(r["errors"]), ";".join(r["warnings"]),
            ])
    return summary


def _main(argv):
    if not argv:
        print(__doc__)
        return 0
    data_dir = argv[0]
    labels = None
    out = "results"
    rest = argv[1:]
    if rest and not rest[0].startswith("--"):
        labels = json.loads(Path(rest[0]).read_text(encoding="utf-8"))
        labels = {k: v for k, v in labels.items() if not k.startswith("_")}
        rest = rest[1:]
    if "--out" in rest:
        out = rest[rest.index("--out") + 1]
    results = validate_dir(data_dir, labels)
    summary = write_reports(results, out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

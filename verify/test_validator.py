"""Unit tests for the npz data validator (synthetic records, no field data)."""
import numpy as np
import pytest

from validate_data import validate_npz, summarize


def _good(path, T=50, D=40, has_hit=True):
    depth = np.linspace(4000, 4000 + 2.0 * (D - 1), D).astype(np.float32)
    rng = np.random.default_rng(0)
    strain = rng.standard_normal((T, D)).astype(np.float32)
    sr = rng.standard_normal((T, D)).astype(np.float32) * 0.1
    np.savez(path, strain=strain, strain_rate=sr, depth=depth,
             dt_sec=np.float32(60.0),
             dist_profile_m=np.abs(rng.standard_normal(D)).astype(np.float32))


def test_good_record_passes(tmp_path):
    p = tmp_path / "CQTW8HAS39F.npz"
    _good(p)
    labels = {"CQTW8HAS39F": {"has_hit": True, "tpk": 30, "T": 50, "D": 40}}
    r = validate_npz(p, labels)
    assert r["ok"], r["errors"]


def test_missing_key_is_error(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, strain=np.zeros((10, 5), np.float32))
    r = validate_npz(p)
    assert not r["ok"]
    assert any(e.startswith("missing_key") for e in r["errors"])


def test_nan_strain_is_error(tmp_path):
    p = tmp_path / "x.npz"
    s = np.zeros((10, 5), np.float32)
    s[3, 2] = np.nan
    np.savez(p, strain=s, strain_rate=np.zeros((10, 5), np.float32),
             depth=np.arange(5).astype(np.float32), dt_sec=np.float32(60.0))
    r = validate_npz(p)
    assert not r["ok"]
    assert any(e.startswith("nonfinite_strain") for e in r["errors"])


def test_shape_mismatch_is_error(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, strain=np.zeros((10, 5), np.float32),
             strain_rate=np.zeros((10, 6), np.float32),
             depth=np.arange(5).astype(np.float32), dt_sec=np.float32(60.0))
    r = validate_npz(p)
    assert not r["ok"]
    assert any(e.startswith("strain_rate_shape") for e in r["errors"])


def test_tpk_out_of_range_is_error(tmp_path):
    p = tmp_path / "H.npz"
    _good(p, T=20, D=10)
    labels = {"H": {"has_hit": True, "tpk": 999, "T": 20, "D": 10}}
    r = validate_npz(p, labels)
    assert not r["ok"]
    assert any(e.startswith("tpk_out_of_range") for e in r["errors"])


def test_dz_out_of_range_is_warning(tmp_path):
    p = tmp_path / "x.npz"
    depth = (np.arange(10) * 50.0).astype(np.float32)  # 50 m spacing
    np.savez(p, strain=np.zeros((8, 10), np.float32),
             strain_rate=np.zeros((8, 10), np.float32),
             depth=depth, dt_sec=np.float32(60.0))
    r = validate_npz(p)
    assert r["ok"]
    assert any(w.startswith("dz_out_of_range") for w in r["warnings"])


def test_summarize_counts(tmp_path):
    good = tmp_path / "g.npz"; _good(good, T=12, D=8)
    bad = tmp_path / "b.npz"; np.savez(bad, strain=np.zeros((4, 3), np.float32))
    results = [validate_npz(good), validate_npz(bad)]
    s = summarize(results)
    assert s["n_files"] == 2 and s["n_ok"] == 1 and s["n_error"] == 1

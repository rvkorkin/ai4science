"""Property-based causality tests for the DAS preprocessing primitives.

Central scientific invariant of the frac-hit pipeline: every preprocessed
sample at (output) time i may depend ONLY on raw input up to real time i*R.
If a future input sample can change a past output, the model trained on that
output has a look-ahead (future-data) leak and its live/offline metrics are
not comparable. This is exactly the failure that retired the per-crop CNN
(see core/WORKLOG.md). These tests verify causality holds for arbitrary
random inputs, decimation factors and shapes.
"""
import numpy as np
from hypothesis import given, settings, strategies as st

import preprocess as pp


def _safe_output_count(k: int, R: int, M: int) -> int:
    """# of leading output rows that depend only on input rows [0 .. k-1].

    Output row i consumes input up to index i*R, so it is unaffected by a
    perturbation at input index >= k iff i*R <= k-1.
    """
    return min(M, (k - 1) // R + 1)


@st.composite
def _decim_case(draw):
    R = draw(st.integers(min_value=1, max_value=8))
    M = draw(st.integers(min_value=2, max_value=25))
    extra = draw(st.integers(min_value=0, max_value=max(0, R - 1)))
    N = M * R + extra
    D = draw(st.integers(min_value=1, max_value=6))
    seed = draw(st.integers(min_value=0, max_value=100_000))
    return R, N, D, seed


@settings(max_examples=250, deadline=None)
@given(_decim_case())
def test_decimate_causal_no_future_leak(case):
    R, N, D, seed = case
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D)).astype(np.float64)

    y1 = pp.decimate_causal(x, R)
    M = y1.shape[0]
    assert M == N // R

    k = int(rng.integers(1, N))
    x2 = x.copy()
    x2[k:] = rng.standard_normal((N - k, D))
    y2 = pp.decimate_causal(x2, R)

    n_safe = _safe_output_count(k, R, M)
    assert np.allclose(y1[:n_safe], y2[:n_safe], atol=1e-9), (
        f"future input at >= {k} changed a causal output (R={R}, N={N})"
    )


@settings(max_examples=120, deadline=None)
@given(_decim_case())
def test_decimate_causal_not_vacuous(case):
    """Perturbing input[0] MUST change output[0] — guards against a test
    that would pass even if the function ignored its input."""
    R, N, D, seed = case
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D)).astype(np.float64) + 1.0
    y1 = pp.decimate_causal(x, R)
    x2 = x.copy()
    x2[0] += 1000.0
    y2 = pp.decimate_causal(x2, R)
    assert not np.allclose(y1[0], y2[0])


@st.composite
def _sr_case(draw):
    T = draw(st.integers(min_value=10, max_value=60))
    D = draw(st.integers(min_value=1, max_value=8))
    dt_sec = draw(st.integers(min_value=10, max_value=60))
    smooth_sec = draw(st.floats(min_value=0.0, max_value=180.0))
    k = draw(st.integers(min_value=1, max_value=T - 1))
    seed = draw(st.integers(min_value=0, max_value=100_000))
    return T, D, dt_sec, smooth_sec, k, seed


@settings(max_examples=100, deadline=None)
@given(_sr_case())
def test_strain_rate_causal_no_future_leak(case):
    T, D, dt_sec, smooth_sec, k, seed = case
    rng = np.random.default_rng(seed)
    strain = rng.standard_normal((T, D)).astype(np.float64)

    y1 = pp.strain_rate_causal(strain, dt_sec=dt_sec, smooth_sec=smooth_sec)
    assert y1.shape == strain.shape
    assert np.allclose(y1[0], 0.0)

    strain2 = strain.copy()
    strain2[k:] = rng.standard_normal((T - k, D))
    y2 = pp.strain_rate_causal(strain2, dt_sec=dt_sec, smooth_sec=smooth_sec)

    assert np.allclose(y1[:k], y2[:k], atol=1e-9), (
        f"strain_rate_causal leaked future (k={k}, smooth={smooth_sec})"
    )


@st.composite
def _raw_case(draw):
    R = draw(st.integers(min_value=2, max_value=6))
    M = draw(st.integers(min_value=3, max_value=20))
    N = M * R + draw(st.integers(min_value=0, max_value=R - 1))
    D = draw(st.integers(min_value=2, max_value=8))
    span = draw(st.integers(min_value=1, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=100_000))
    return R, N, D, span, seed


@settings(max_examples=200, deadline=None)
@given(_raw_case())
def test_strain_rate_from_raw_no_future_leak(case):
    R, N, D, span, seed = case
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((N, D)).astype(np.float32)

    s1, sr1 = pp.strain_rate_from_raw(raw, dt_in_sec=10.0, R=R, span_raw=span,
                                      dz_m=2.0, sigma_m=2.0)
    M = s1.shape[0]
    assert M == N // R
    assert sr1.shape == s1.shape

    k = int(rng.integers(1, N))
    raw2 = raw.copy()
    raw2[k:] = rng.standard_normal((N - k, D)).astype(np.float32)
    s2, sr2 = pp.strain_rate_from_raw(raw2, dt_in_sec=10.0, R=R, span_raw=span,
                                      dz_m=2.0, sigma_m=2.0)

    n_safe = _safe_output_count(k, R, M)
    assert np.allclose(s1[:n_safe], s2[:n_safe], atol=1e-4), "strain leaked future"
    assert np.allclose(sr1[:n_safe], sr2[:n_safe], atol=1e-4), "strain_rate leaked future"


@settings(max_examples=60, deadline=None)
@given(_decim_case())
def test_decimate_causal_deterministic(case):
    R, N, D, seed = case
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D)).astype(np.float64)
    assert np.array_equal(pp.decimate_causal(x, R), pp.decimate_causal(x, R))


# ─────────────────────────────────────────────────────────────────────────────
#  strain_rate_causal: y[0]==0 + non-vacuous
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, deadline=None)
@given(_sr_case())
def test_strain_rate_causal_y0_is_zero(case):
    """y[0] must be exactly 0 — no data exists before the recording starts."""
    T, D, dt_sec, smooth_sec, k, seed = case
    rng = np.random.default_rng(seed)
    strain = rng.standard_normal((T, D)).astype(np.float64)
    y = pp.strain_rate_causal(strain, dt_sec=dt_sec, smooth_sec=smooth_sec)
    assert np.allclose(y[0], 0.0, atol=1e-8), "y[0] != 0"


@settings(max_examples=100, deadline=None)
@given(_sr_case())
def test_strain_rate_causal_not_vacuous(case):
    """Perturbing strain[k] MUST change y[k] — otherwise the test is meaningless."""
    T, D, dt_sec, smooth_sec, k, seed = case
    rng = np.random.default_rng(seed)
    strain = rng.standard_normal((T, D)).astype(np.float64)
    y1 = pp.strain_rate_causal(strain, dt_sec=dt_sec, smooth_sec=smooth_sec)

    strain2 = strain.copy()
    strain2[k] += 1000.0
    y2 = pp.strain_rate_causal(strain2, dt_sec=dt_sec, smooth_sec=smooth_sec)

    assert not np.allclose(y1[k], y2[k], atol=1e-8), (
        f"perturbation at strain[{k}] did NOT change y[{k}] — vacuous test"
    )

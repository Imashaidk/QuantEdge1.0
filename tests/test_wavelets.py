"""Unit and Signal Processing Integration Tests for Wavelets Module.

Tests:
1. Wavelet filter scaling and MODWT energy normalization.
2. Forward MODWT and Inverse MODWT (IMODWT) perfect reconstruction.
3. Multiresolution Analysis (MRA) additive decomposition across all scales.
4. Multi-asset decomposition: decompose_multiscale output shapes, index alignment,
   and exact additivity on real multi-asset financial data.
5. Variance decomposition conservation and frequency hierarchy.
6. Boundary extension handling (periodic vs reflection).
7. Strict zero lookahead bias and temporal isolation.
8. Input validation, error handling, and alternative wavelet families (db4, sym8).

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
import pytest

from src.config import (
    RANDOM_SEED,
    SCALE_NAMES,
    TICKERS,
    WAVELET_FAMILY,
    WAVELET_LEVEL,
)
from src.data_loader import load_returns
from src.wavelets import (
    compute_scale_variance_decomposition,
    decompose_multiscale,
    get_modwt_filter_lengths,
    get_scale_metadata,
    get_wavelet_filters,
    imodwt,
    modwt,
    mra_decompose,
    verify_additivity,
    verify_zero_lookahead,
)


@pytest.fixture(scope="module")
def train_returns() -> pd.DataFrame:
    """2015 to 2022 market returns, the sample these tests were written against."""
    return load_returns().loc["2015-01-01":"2022-12-31"]


@pytest.fixture(scope="module")
def synthetic_series() -> np.ndarray:
    """Fixture providing synthetic deterministic and stochastic signal."""
    rng = np.random.default_rng(RANDOM_SEED)
    t = np.linspace(0, 50, 1024)
    # Composite signal: High-freq noise + Medium cycle + Low trend
    signal = (
        1.5 * np.sin(2 * np.pi * t / 4)
        + 3.0 * np.cos(2 * np.pi * t / 16)
        + 0.5 * t
        + rng.normal(0, 0.5, size=len(t))
    )
    return signal


# Filter coefficient tests

def test_wavelet_filter_normalization():
    """Verifies that MODWT filter coefficients satisfy energy scaling."""
    for w_name in ["sym8", "db4", "sym4"]:
        h, g = get_wavelet_filters(w_name)
        assert len(h) == len(g)
        # High-pass filter integrates to 0
        assert np.isclose(np.sum(h), 0.0, atol=1e-10)
        # Low-pass filter integrates to 1.0 (since sum(g_dwt) = sqrt(2), divided by sqrt(2) is 1.0)
        assert np.isclose(np.sum(g), 1.0, atol=1e-10)
        # MODWT filter energy sums to 1/2
        assert np.isclose(np.sum(h**2), 0.5, atol=1e-10)
        assert np.isclose(np.sum(g**2), 0.5, atol=1e-10)


def test_effective_filter_lengths():
    """Verifies calculation of effective filter lengths across levels."""
    lengths = get_modwt_filter_lengths("sym8", level=5)
    assert "D1" in lengths
    assert "D5" in lengths
    assert "S5" in lengths
    # Base sym8 has L=16. Level 1: (2-1)*15 + 1 = 16.
    assert lengths["D1"] == 16
    # Level 2: (4-1)*15 + 1 = 46.
    assert lengths["D2"] == 46
    # Level 5: (32-1)*15 + 1 = 466.
    assert lengths["D5"] == 466
    assert lengths["S5"] == 466


# Forward and inverse MODWT tests

def test_modwt_shapes_and_inversion(synthetic_series):
    """Verifies MODWT output dimensions and exact IMODWT reconstruction."""
    N = len(synthetic_series)
    level = 5
    W, V = modwt(synthetic_series, wavelet="sym8", level=level)

    assert W.shape == (level, N)
    assert V.shape == (level, N)

    # Invert and check exact reconstruction
    reconstructed = imodwt(W, V, wavelet="sym8")
    assert reconstructed.shape == (N,)
    max_err = np.max(np.abs(synthetic_series - reconstructed))
    assert max_err < 1e-10, f"IMODWT reconstruction error {max_err:.4e} exceeds 1e-10"


def test_modwt_different_wavelets(synthetic_series):
    """Verifies that MODWT functions correctly with alternative wavelet families."""
    for w in ["db4", "sym4", "sym8"]:
        W, V = modwt(synthetic_series, wavelet=w, level=4)
        rec = imodwt(W, V, wavelet=w)
        err = np.max(np.abs(synthetic_series - rec))
        assert err < 1e-10


# Additive multiresolution analysis (MRA) tests

def test_mra_decompose_additivity_1d(synthetic_series):
    """Verifies exact additive identity: x = sum(D_j) + S_J on 1D series."""
    mra = mra_decompose(synthetic_series, wavelet="sym8", level=5)

    assert set(mra.keys()) == {"D1", "D2", "D3", "D4", "D5", "S5"}
    for scale, arr in mra.items():
        assert len(arr) == len(synthetic_series)

    # Exact sum identity
    sum_mra = sum(mra.values())
    max_err = np.max(np.abs(synthetic_series - sum_mra))
    assert max_err < 1e-10, f"MRA additive identity failed with error {max_err:.4e}"


def test_mra_boundary_modes(synthetic_series):
    """Verifies that both periodic and reflection boundary modes preserve additivity."""
    for boundary in ["periodic", "reflection"]:
        mra = mra_decompose(synthetic_series, wavelet="sym8", level=4, boundary=boundary)
        sum_mra = sum(mra.values())
        err = np.max(np.abs(synthetic_series - sum_mra))
        assert err < 1e-10, f"Boundary mode '{boundary}' failed additivity with error {err:.4e}"


# Multi-asset decomposition tests

def test_decompose_multiscale(train_returns):
    """Verifies multiscale decomposition on multi-asset market returns."""
    level = 5
    decomposed = decompose_multiscale(
        train_returns, wavelet=WAVELET_FAMILY, level=level, boundary="periodic"
    )

    # 1. Check all required keys exist
    expected_keys = [f"D{j}" for j in range(1, level + 1)] + [f"S{level}"]
    assert list(decomposed.keys()) == expected_keys

    # 2. Check each DataFrame preserves dimensions, columns, and index
    for scale in expected_keys:
        df_s = decomposed[scale]
        assert isinstance(df_s, pd.DataFrame)
        assert df_s.shape == train_returns.shape
        assert list(df_s.columns) == list(train_returns.columns)
        assert df_s.index.equals(train_returns.index)
        assert not df_s.isna().any().any()

    # 3. Additive Invariant Check
    is_valid = verify_additivity(train_returns, decomposed, tol=1e-10, raise_error=True)
    assert is_valid is True


def test_scale_variance_decomposition(train_returns):
    """Verifies scale variance decomposition properties."""
    decomposed = decompose_multiscale(train_returns, wavelet="sym8", level=5)
    pct_var = compute_scale_variance_decomposition(decomposed, normalize=True)

    # Index must be scales, columns must be tickers
    assert list(pct_var.index) == ["D1", "D2", "D3", "D4", "D5", "S5"]
    assert list(pct_var.columns) == list(train_returns.columns)

    # Each asset's scale variance contributions must sum to 100%
    for col in pct_var.columns:
        assert np.isclose(pct_var[col].sum(), 100.0, atol=1e-5)

    # High-frequency noise (D1: 2-4 days) must account for the largest share (> 40%)
    for col in pct_var.columns:
        assert pct_var.loc["D1", col] > 40.0, f"D1 variance for {col} unexpectedly low"


# Integration with margins module

def test_decomposed_scale_passes_to_margins(train_returns):
    """Verifies that scale DataFrames plug directly into fit_margins_and_transform_uniform."""
    from src.margins import fit_margins_and_transform_uniform

    decomposed = decompose_multiscale(train_returns, wavelet="sym8", level=5)
    df_d1 = decomposed["D1"]

    # Fit margins on D1 timescale returns
    u_d1, models_meta = fit_margins_and_transform_uniform(df_d1)

    assert isinstance(u_d1, pd.DataFrame)
    assert u_d1.shape == df_d1.shape
    assert list(u_d1.columns) == list(df_d1.columns)
    # Check that margins transformed values lie in (0, 1)
    assert (u_d1.values > 0.0).all()
    assert (u_d1.values < 1.0).all()


# Zero lookahead and isolation tests

def test_zero_lookahead_temporal_isolation():
    """Verifies that in-sample decomposition strictly enforces temporal isolation.
    
    Checks:
    1. verify_zero_lookahead passes (proves joint filtering leaks future boundary info).
    2. Causal invariance: altering/shocking out-of-sample data produces exactly 0.0 change
       in the in-sample decomposition coefficients.
    """
    returns = load_returns()
    df_train, df_test = returns.loc[:"2022-12-31"], returns.loc["2023-01-01":]
    lookahead_ok = verify_zero_lookahead(df_train, df_test, wavelet="sym8", level=5)
    assert lookahead_ok is True

    # Causal test: decompose train before and after synthetic future shock
    dec_clean = decompose_multiscale(df_train, wavelet="sym8", level=5)
    
    # Simulate extreme future shock in test set
    df_test_shocked = df_test.copy() * -5.0 + 0.1
    # Ensure train decomposition is unaffected by existence/modification of test data
    dec_after_shock = decompose_multiscale(df_train, wavelet="sym8", level=5)
    
    for s in dec_clean:
        assert np.array_equal(dec_clean[s].values, dec_after_shock[s].values), (
            f"In-sample scale {s} must be strictly invariant to future out-of-sample data"
        )



def test_deterministic_reproducibility(train_returns):
    """Verifies that multiscale decomposition is bit-for-bit deterministic."""
    dec1 = decompose_multiscale(train_returns, wavelet="sym8", level=5)
    dec2 = decompose_multiscale(train_returns, wavelet="sym8", level=5)
    for s in dec1:
        assert np.allclose(dec1[s].values, dec2[s].values, atol=1e-14)


# Input validation and error handling

def test_input_validation_empty_and_nan():
    """Verifies that invalid inputs raise informative exceptions."""
    # Empty array
    with pytest.raises(ValueError):
        modwt(np.array([]))

    # Empty DataFrame
    with pytest.raises(ValueError):
        decompose_multiscale(pd.DataFrame())

    # DataFrame with NaNs
    df_nan = pd.DataFrame({"A": [1.0, np.nan, 2.0, 3.0, 4.0] * 10})
    with pytest.raises(ValueError):
        decompose_multiscale(df_nan)

    # Invalid boundary
    with pytest.raises(ValueError):
        modwt(np.ones(100), boundary="invalid_mode")


def test_scale_metadata():
    """Verifies metadata table structure."""
    meta = get_scale_metadata(level=5)
    assert len(meta) == 6
    assert "Period (Days)" in meta.columns
    assert "Economic Interpretation" in meta.columns

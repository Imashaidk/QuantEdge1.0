"""Unit and Econometric Integration Tests for Margins and Copulas Modules.

Tests:
1. AR(1)-GJR-GARCH(1,1) + EVT-POT marginal estimation and PIT calibration.
2. Kolmogorov-Smirnov uniformity validation (p > 0.05).
3. Exact CDF-PPF invertibility and numerical stability.
4. Scale-Optimal Copula Tournament across Gaussian, Student-t, Clayton, Gumbel, Frank.
5. Timescale Asymmetry Ratio (TAR) calculation and theoretical tail bounds.
6. Synthetic joint return simulation and empirical cross-asset dependence preservation.
7. Strict seed determinism (SEED = 42).
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

from src.config import RANDOM_SEED, TICKERS
from src.copulas import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    compute_empirical_tail_dependence,
    compute_pairwise_tail_matrix,
    run_scale_copula_tournament,
    simulate_copula_joint_returns,
)
from src.data_loader import load_and_split_data
from src.margins import GARCH_EVT_Margin, fit_margins_and_transform_uniform


@pytest.fixture(scope="module")
def real_data():
    """Loads in-sample market return data for testing."""
    df_train, _ = load_and_split_data()
    return df_train


@pytest.fixture(scope="module")
def fitted_margins(real_data):
    """Fits margins once for the module tests."""
    u_df, models_meta = fit_margins_and_transform_uniform(real_data)
    return u_df, models_meta


# ==============================================================================
# 1. MARGINAL GARCH-EVT TESTS
# ==============================================================================

def test_single_margin_garch_evt_fit(real_data):
    """Verifies that GARCH_EVT_Margin fits correctly on SPY return series."""
    spy_series = real_data["SPY"]
    margin = GARCH_EVT_Margin(asset_name="SPY", tail_percentile=0.10)
    margin.fit(spy_series)

    # Threshold checks
    assert margin.u_L < margin.u_U, "Lower threshold must be strictly below upper threshold."
    assert margin.u_L < 0.0, "Lower tail threshold for standardized residuals should be negative."
    assert margin.u_U > 0.0, "Upper tail threshold for standardized residuals should be positive."

    # GPD parameter checks
    assert -0.5 <= margin.xi_L <= 0.8, f"Lower tail index xi_L ({margin.xi_L}) outside valid domain."
    assert -0.5 <= margin.xi_U <= 0.8, f"Upper tail index xi_U ({margin.xi_U}) outside valid domain."
    assert margin.beta_L > 0, "Lower GPD scale parameter must be positive."
    assert margin.beta_U > 0, "Upper GPD scale parameter must be positive."

    # Kolmogorov-Smirnov test for uniformity of PIT residuals
    assert margin.ks_pvalue > 0.05, f"PIT failed KS uniformity test: p = {margin.ks_pvalue:.4f} <= 0.05"


def test_margin_invertibility_roundtrip(fitted_margins):
    """Verifies that CDF and PPF (Quantile function) are exact inverse bijections."""
    _, models_meta = fitted_margins
    test_u = np.linspace(0.0001, 0.9999, 500)

    for ticker in ["SPY", "QQQ", "TLT", "GLD", "HYG"]:
        model = models_meta["models"][ticker]
        z = model.ppf(test_u)
        u_rec = model.cdf(z)
        max_error = np.max(np.abs(test_u - u_rec))
        assert max_error < 1e-6, f"CDF-PPF roundtrip error too large for {ticker}: {max_error:.4e}"


def test_fit_margins_and_transform_uniform_contract(real_data, fitted_margins):
    """Verifies Contract 3 interface and data invariants."""
    u_df, models_meta = fitted_margins

    # Dimensional invariants
    assert u_df.shape == real_data.shape, "Transformed margins must preserve exact shape."
    assert list(u_df.columns) == list(real_data.columns), "Columns must match original assets."
    assert (u_df.index == real_data.index).all(), "DatetimeIndex must match exactly."

    # Bounded domain: U in (0, 1)
    assert (u_df.values > 0.0).all(), "All uniform margins must be strictly positive."
    assert (u_df.values < 1.0).all(), "All uniform margins must be strictly less than 1."
    assert not np.isnan(u_df.values).any(), "No NaN values allowed in uniform margins."


# ==============================================================================
# 2. INDIVIDUAL COPULA FAMILY TESTS
# ==============================================================================

def test_gaussian_copula(fitted_margins):
    """Verifies Gaussian copula estimation, zero tail dependence, and simulation."""
    u_df, _ = fitted_margins
    cop = GaussianCopula()
    cop.fit(u_df.values)

    assert cop.log_likelihood > 0, "Gaussian copula log-likelihood should be positive on correlated assets."
    assert np.isfinite(cop.aic) and np.isfinite(cop.bic)
    assert cop.lambda_L == 0.0, "Gaussian copula must have zero theoretical lower tail dependence."
    assert cop.lambda_U == 0.0, "Gaussian copula must have zero theoretical upper tail dependence."
    assert cop.tar == 0.0, "Gaussian copula TAR must be zero."

    samples = cop.sample(n_samples=500, seed=42)
    assert samples.shape == (500, 5)
    assert (samples >= 0.0).all() and (samples <= 1.0).all()


def test_student_t_copula(fitted_margins):
    """Verifies Student-t copula estimation, degrees of freedom, and symmetric tail dependence."""
    u_df, _ = fitted_margins
    cop = StudentTCopula()
    cop.fit(u_df.values)

    assert 2.1 <= cop.nu <= 40.0, f"Estimated nu ({cop.nu:.2f}) outside acceptable range."
    assert cop.log_likelihood > 0
    assert cop.lambda_L > 0.0, "Student-t copula on equities must have positive tail dependence."
    assert cop.lambda_L == cop.lambda_U, "Student-t copula must have symmetric tail dependence."
    assert cop.tar == 0.0, "Student-t copula TAR must be zero."

    samples = cop.sample(n_samples=500, seed=42)
    assert samples.shape == (500, 5)


def test_clayton_copula(fitted_margins):
    """Verifies Clayton copula asymmetric lower-tail crash dependence."""
    u_df, _ = fitted_margins
    cop = ClaytonCopula()
    cop.fit(u_df.values)

    assert cop.theta > 0, "Clayton theta must be positive."
    assert cop.lambda_L > 0.0, "Clayton lower tail dependence must be positive."
    assert cop.lambda_U == 0.0, "Clayton upper tail dependence must be strictly zero."
    assert cop.tar == cop.lambda_L, "Clayton TAR must equal lambda_L."

    samples = cop.sample(n_samples=500, seed=42)
    assert samples.shape == (500, 5)


def test_gumbel_copula(fitted_margins):
    """Verifies Gumbel copula asymmetric upper-tail boom dependence."""
    u_df, _ = fitted_margins
    cop = GumbelCopula()
    cop.fit(u_df.values)

    assert cop.theta >= 1.0, "Gumbel theta must be >= 1.0."
    assert cop.lambda_U > 0.0, "Gumbel upper tail dependence must be positive."
    assert cop.lambda_L == 0.0, "Gumbel lower tail dependence must be strictly zero."
    assert cop.tar < 0.0, "Gumbel TAR must be negative."

    samples = cop.sample(n_samples=500, seed=42)
    assert samples.shape == (500, 5)


def test_frank_copula(fitted_margins):
    """Verifies Frank copula radial symmetry and zero tail dependence."""
    u_df, _ = fitted_margins
    cop = FrankCopula()
    cop.fit(u_df.values)

    assert cop.lambda_L == 0.0
    assert cop.lambda_U == 0.0
    assert cop.tar == 0.0

    samples = cop.sample(n_samples=500, seed=42)
    assert samples.shape == (500, 5)


# ==============================================================================
# 3. SCALE-OPTIMAL TOURNAMENT & SIMULATION TESTS
# ==============================================================================

def test_copula_tournament_selection(fitted_margins):
    """Verifies that run_scale_copula_tournament selects a valid winner and returns all metrics."""
    u_df, _ = fitted_margins
    res = run_scale_copula_tournament(u_df, scale_name="D1_Test")

    # Contract 3 dictionary keys check
    expected_keys = [
        "scale",
        "best_copula",
        "lambda_L",
        "lambda_U",
        "tar",
        "bic_scores",
        "aic_scores",
        "log_likelihoods",
        "fitted_copula_obj",
        "all_models",
        "empirical_tail_dep",
    ]
    for key in expected_keys:
        assert key in res, f"Tournament result missing required contract key: {key}"

    assert res["scale"] == "D1_Test"
    assert res["best_copula"] in ["gaussian", "student_t", "clayton", "gumbel", "frank"]
    assert res["lambda_L"] >= 0.0
    assert res["lambda_U"] >= 0.0

    # Ensure best copula corresponds to lowest BIC score
    best_name = res["best_copula"]
    assert res["bic_scores"][best_name] == min(res["bic_scores"].values())


def test_pairwise_tail_dependence_matrix(fitted_margins):
    """Verifies pairwise empirical tail matrix calculation."""
    u_df, _ = fitted_margins
    pairwise_df = compute_pairwise_tail_matrix(u_df, q=0.05)

    assert len(pairwise_df) == 10  # 5 * 4 / 2 = 10 pairs
    assert "lambda_L" in pairwise_df.columns
    assert "lambda_U" in pairwise_df.columns
    assert "TAR" in pairwise_df.columns

    # High equity co-crash between SPY and QQQ
    spy_qqq_row = pairwise_df[
        (pairwise_df["Asset_1"] == "SPY") & (pairwise_df["Asset_2"] == "QQQ")
    ].iloc[0]
    assert spy_qqq_row["lambda_L"] > 0.40, "SPY and QQQ should exhibit high crash tail dependence (>0.40)."


def test_simulate_copula_joint_returns(fitted_margins):
    """Verifies that simulate_copula_joint_returns generates correct distribution."""
    u_df, models_meta = fitted_margins
    res = run_scale_copula_tournament(u_df, scale_name="InSample")

    sim_df = simulate_copula_joint_returns(
        fitted_copula_obj=res["fitted_copula_obj"],
        models_meta=models_meta,
        n_samples=5000,
        seed=RANDOM_SEED,
    )

    assert sim_df.shape == (5000, 5)
    assert list(sim_df.columns) == TICKERS
    assert not sim_df.isna().any().any()

    # Economic validation: SPY and QQQ should remain strongly positively correlated
    sim_corr = sim_df.corr()
    assert sim_corr.loc["SPY", "QQQ"] > 0.80, "Simulated SPY-QQQ correlation must be high (>0.80)."
    # TLT and SPY should have negative/low correlation
    assert sim_corr.loc["SPY", "TLT"] < 0.10, "Simulated SPY-TLT correlation should reflect diversification."


def test_deterministic_simulation_seed(fitted_margins):
    """Verifies that simulation is strictly deterministic given the same seed."""
    u_df, models_meta = fitted_margins
    res = run_scale_copula_tournament(u_df, scale_name="InSample")

    sim_1 = simulate_copula_joint_returns(res["fitted_copula_obj"], models_meta, n_samples=1000, seed=42)
    sim_2 = simulate_copula_joint_returns(res["fitted_copula_obj"], models_meta, n_samples=1000, seed=42)

    np.testing.assert_allclose(
        sim_1.values,
        sim_2.values,
        err_msg="Simulation results with same seed must be identical bit-for-bit.",
    )

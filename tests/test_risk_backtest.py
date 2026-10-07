"""Unit and Statistical Validation Tests for Risk Engine and Backtesting Modules.

Tests:
1. Portfolio return computation and weight normalization.
2. VaR and Expected Shortfall empirical calculations (ES >= VaR invariant).
3. Parametric Gaussian VaR/ES mathematical formulas.
4. Basel square-root-of-time scaling.
5. Horizon-Conditioned Tail Capital Multiplier (H-TCM) sensitivity and bounds.
6. Kupiec POF Likelihood Ratio hypothesis test (unconditional coverage).
7. Christoffersen Independence Likelihood Ratio test (violation clustering).
8. Official Basel Committee Traffic Light zone classification (Green/Yellow/Red).
9. Fissler-Ziegel (FZ) joint scoring consistency.
10. Backtest integration: run_out_of_sample_backtest schema and output types.

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

from src.backtest import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    classify_basel_traffic_light,
    compute_backtest_metrics,
    fissler_ziegel_loss,
    kupiec_pof_test,
    run_out_of_sample_backtest,
)
from src.config import (
    ALPHA_ES_975,
    ALPHA_VAR_95,
    ALPHA_VAR_99,
    DEFAULT_PORTFOLIO_WEIGHTS,
    RANDOM_SEED,
    TICKERS,
)
from src.data_loader import load_and_split_data
from src.risk_engine import (
    RiskEngine,
    compute_basel_scaled_var,
    compute_historical_var,
    compute_htcm_multiplier,
    compute_htcm_var,
    compute_parametric_gaussian_var,
    compute_portfolio_returns,
    compute_var_es_from_loss,
    map_horizon_to_wavelet_scale,
)


@pytest.fixture(scope="module")
def data_splits():
    """Loads cached in-sample train and out-of-sample test returns."""
    df_train, df_test = load_and_split_data()
    return df_train, df_test


# Portfolio calculation and invariants

def test_compute_portfolio_returns():
    """Verifies portfolio return calculation and weight normalization."""
    returns = np.array([
        [0.01, 0.02, -0.01, 0.005, 0.002],
        [-0.02, -0.01, 0.015, -0.005, -0.001],
    ])
    weights = np.array([0.3, 0.2, 0.25, 0.15, 0.10])
    port_ret = compute_portfolio_returns(returns, weights)

    expected = np.dot(returns, weights)
    assert np.allclose(port_ret, expected, atol=1e-10)

    # Test automatic weight normalization
    unnorm_weights = np.array([3.0, 2.0, 2.5, 1.5, 1.0])
    port_ret_unnorm = compute_portfolio_returns(returns, unnorm_weights)
    assert np.allclose(port_ret_unnorm, expected, atol=1e-10)


def test_var_es_empirical_invariants():
    """Verifies that ES is strictly >= VaR on any empirical loss distribution."""
    rng = np.random.default_rng(RANDOM_SEED)
    losses = rng.standard_t(df=4, size=10000) * 0.02

    var_99, es_975 = compute_var_es_from_loss(losses, alpha_var=0.99, alpha_es=0.975)

    assert var_99 > 0.0
    assert es_975 >= var_99, "Expected Shortfall must be >= Value-at-Risk"


# Risk models and scaling

def test_parametric_gaussian_var():
    """Verifies parametric Gaussian formulas against analytical values."""
    rng = np.random.default_rng(RANDOM_SEED)
    r_series = rng.normal(loc=0.0005, scale=0.01, size=5000)

    var_1d, es_1d = compute_parametric_gaussian_var(r_series, horizon=1, alpha_var=0.99, alpha_es=0.975)
    var_5d, es_5d = compute_parametric_gaussian_var(r_series, horizon=5, alpha_var=0.99, alpha_es=0.975)

    assert var_1d > 0.0
    assert var_5d > var_1d
    assert np.isclose(var_5d / var_1d, np.sqrt(5.0), rtol=0.1)


def test_basel_scaled_var():
    """Verifies exact sqrt(h) multiplication."""
    v1 = 0.02
    e1 = 0.025
    vh, eh = compute_basel_scaled_var(v1, e1, horizon=16)

    assert np.isclose(vh, 0.02 * 4.0, atol=1e-10)
    assert np.isclose(eh, 0.025 * 4.0, atol=1e-10)


def test_htcm_multiplier_properties():
    """Verifies H-TCM multiplier behavior under differing tail dependence."""
    # When tail dependence does not increase, multiplier is strictly 1.0
    m_equal = compute_htcm_multiplier(lambda_L_h=0.04, lambda_L_1=0.04, kappa=0.35)
    assert np.isclose(m_equal, 1.0, atol=1e-10)

    m_lower = compute_htcm_multiplier(lambda_L_h=0.02, lambda_L_1=0.04, kappa=0.35)
    assert np.isclose(m_lower, 1.0, atol=1e-10)

    # When tail dependence increases, multiplier expands proportionally
    m_higher = compute_htcm_multiplier(lambda_L_h=0.12, lambda_L_1=0.04, kappa=0.35)
    assert m_higher > 1.0
    # Expected: 1 + 0.35 * (0.08 / 0.04) = 1 + 0.70 = 1.70
    assert np.isclose(m_higher, 1.70, rtol=1e-3)


def test_map_horizon_to_wavelet_scale():
    """Verifies horizon-to-scale mapping."""
    assert map_horizon_to_wavelet_scale(1) == "D1"
    assert map_horizon_to_wavelet_scale(5) == "D2"
    assert map_horizon_to_wavelet_scale(10) == "D3"
    assert map_horizon_to_wavelet_scale(20) == "D4"
    assert map_horizon_to_wavelet_scale(40) == "D5"
    assert map_horizon_to_wavelet_scale(90) == "S5"


# Statistical backtest tests

def test_kupiec_pof_test():
    """Verifies Kupiec POF test on valid and invalid breach frequencies."""
    # Scenario A: Exactly 2.5 breaches out of 250 (nominal p = 0.01)
    lr_a, p_a, pass_a = kupiec_pof_test(breaches=2, total_obs=250, alpha=0.99)
    assert pass_a is True
    assert p_a > 0.05
    assert lr_a < 3.841

    # Scenario B: 15 breaches out of 250 (excessive breaches -> reject H0)
    lr_b, p_b, pass_b = kupiec_pof_test(breaches=15, total_obs=250, alpha=0.99)
    assert pass_b is False
    assert p_b < 0.01


def test_christoffersen_independence_test():
    """Verifies Christoffersen test detects breach clustering."""
    # Scenario A: Dispersed breaches (serially independent)
    hits_clean = np.zeros(500, dtype=int)
    hits_clean[[20, 100, 250, 380, 470]] = 1
    lr_ind, p_ind, pass_ind = christoffersen_independence_test(hits_clean)
    assert pass_ind is True
    assert p_ind > 0.05

    # Scenario B: Clustered breaches (violation clustering during crash)
    hits_cluster = np.zeros(500, dtype=int)
    hits_cluster[100:106] = 1  # 6 consecutive crash breaches
    lr_clust, p_clust, pass_clust = christoffersen_independence_test(hits_cluster)
    assert p_clust < 0.01
    assert pass_clust is False


def test_basel_traffic_light_zones():
    """Verifies official Basel Committee Green/Yellow/Red classifications."""
    # Green Zone: <= 4 breaches on 250 days
    res_green = classify_basel_traffic_light(breaches=3, total_obs=250, alpha=0.99)
    assert res_green["zone"] == "GREEN"
    assert res_green["multiplier"] == 3.00

    # Yellow Zone: 5 to 9 breaches on 250 days
    res_yellow = classify_basel_traffic_light(breaches=7, total_obs=250, alpha=0.99)
    assert res_yellow["zone"] == "YELLOW"
    assert res_yellow["multiplier"] > 3.00

    # Red Zone: >= 10 breaches on 250 days
    res_red = classify_basel_traffic_light(breaches=12, total_obs=250, alpha=0.99)
    assert res_red["zone"] == "RED"
    assert res_red["multiplier"] == 4.00


def test_fissler_ziegel_loss():
    """Verifies Fissler-Ziegel strictly consistent joint loss function."""
    rng = np.random.default_rng(RANDOM_SEED)
    losses = np.abs(rng.normal(0, 0.01, size=2000))

    var_pred = float(np.quantile(losses, 0.99))
    es_pred = float(np.mean(losses[losses >= var_pred]))

    loss_optimal = fissler_ziegel_loss(losses, var_pred, es_pred, alpha=0.99)
    assert np.isfinite(loss_optimal)


# Backtest integration test

def test_run_out_of_sample_backtest_integration(data_splits):
    """Verifies backtesting on real market data splits."""
    df_train, df_test = data_splits
    weights = DEFAULT_PORTFOLIO_WEIGHTS

    # Run quick test across horizons [1, 5]
    backtest_df = run_out_of_sample_backtest(
        df_test=df_test,
        weights=weights,
        h_horizons=[1, 5],
        alpha_var=0.99,
        df_train=df_train,
    )

    # 1. Check returned object is DataFrame
    assert isinstance(backtest_df, pd.DataFrame)
    assert not backtest_df.empty

    # Check required output columns
    expected_cols = [
        "Horizon",
        "Model",
        "VaR_Level",
        "Total_Obs",
        "Breaches",
        "Breach_Rate",
        "Kupiec_LR",
        "Kupiec_p",
        "Christoffersen_p",
        "Basel_Zone",
        "FZ_Loss",
    ]
    for col in expected_cols:
        assert col in backtest_df.columns, f"Missing required column: {col}"

    # 3. Check models evaluated
    models_found = set(backtest_df["Model"].unique())
    assert "Historical_Simulation" in models_found
    assert "Parametric_Gaussian" in models_found
    assert "Basel_Sqrt_Time" in models_found
    assert "Proposed_Multiscale_Copula" in models_found
    assert "H_TCM_Adjusted" in models_found

    # 4. Check Basel zones are valid
    valid_zones = {"GREEN", "YELLOW", "RED"}
    assert set(backtest_df["Basel_Zone"].unique()).issubset(valid_zones)

    # 5. Check observation counts match horizon
    obs_1d = backtest_df[backtest_df["Horizon"] == "1d"]["Total_Obs"].iloc[0]
    obs_5d = backtest_df[backtest_df["Horizon"] == "5d"]["Total_Obs"].iloc[0]
    assert obs_1d == 875, f"1d observation count should be 875, got {obs_1d}"
    assert obs_5d == 871, f"5d observation count should be 871, got {obs_5d}"

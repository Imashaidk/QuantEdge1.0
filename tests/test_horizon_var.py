"""Tests for the horizon-aware VaR model.

Author: Sameera Ekanayaka
"""

import numpy as np
import pandas as pd
import pytest

from src.horizon_var import AssetGarch, HorizonVaRModel, benchmark_forecasts, measured_risk_table


@pytest.fixture
def window():
    rng = np.random.default_rng(7)
    cov = np.array([
        [1.0, 0.9, -0.3, 0.0, 0.7],
        [0.9, 1.0, -0.2, 0.0, 0.6],
        [-0.3, -0.2, 1.0, 0.3, -0.1],
        [0.0, 0.0, 0.3, 1.0, 0.1],
        [0.7, 0.6, -0.1, 0.1, 1.0],
    ])
    x = rng.multivariate_normal(np.zeros(5), cov, size=800) * 0.01
    x = x * rng.standard_t(5, size=(800, 1)) / 1.3
    idx = pd.date_range("2018-01-01", periods=800, freq="B")
    return pd.DataFrame(x, index=idx, columns=["SPY", "QQQ", "TLT", "GLD", "HYG"])


def test_garch_horizon_variance_reverts_to_long_run():
    g = AssetGarch(mu=0.0, omega=0.05, alpha=0.05, gamma=0.05, beta=0.85, last_sigma2=4.0, last_eps=0.0)
    long_run = 0.05 / (1 - 0.05 - 0.025 - 0.85)
    # Starting above the long-run level, the h-day variance grows slower than h times today's.
    assert g.horizon_variance(20) < 20 * g.horizon_variance(1)
    assert g.horizon_variance(2000) / 2000 == pytest.approx(long_run, rel=0.05)


def test_garch_state_update_uses_current_residual():
    """Verifies that AssetGarch.update uses the newly observed residual, not the lagged one."""
    mu = 0.02
    omega = 0.05
    alpha = 0.08
    gamma = 0.12
    beta = 0.82
    prior_sigma2 = 2.5
    old_eps = 0.3

    g = AssetGarch(mu=mu, omega=omega, alpha=alpha, gamma=gamma, beta=beta,
                   last_sigma2=prior_sigma2, last_eps=old_eps)

    new_r = -1.5
    new_eps = new_r - mu
    assert new_eps < 0.0

    expected_sigma2 = omega + (alpha + gamma) * (new_eps ** 2) + beta * prior_sigma2
    buggy_sigma2 = omega + alpha * (old_eps ** 2) + beta * prior_sigma2

    g.update(new_r)

    assert g.last_sigma2 == pytest.approx(expected_sigma2, rel=1e-6)
    assert g.last_eps == pytest.approx(new_eps, rel=1e-6)
    assert abs(g.last_sigma2 - buggy_sigma2) > 0.1


def test_one_day_models_agree(window):
    model = HorizonVaRModel.fit(window, [1, 5], n_sims=4000)
    fc = model.forecast(1, 0.99)
    assert fc["daily_sqrt"] == pytest.approx(fc["daily_copula"])
    assert fc["daily_copula"] == pytest.approx(fc["horizon_copula"])


def test_es_not_below_var(window):
    model = HorizonVaRModel.fit(window, [5], n_sims=4000)
    for var, es in model.forecast(5, 0.99).values():
        assert es >= var > 0


def test_benchmarks_scale_with_horizon(window):
    port = window.mean(axis=1).to_numpy()
    one = benchmark_forecasts(port, 1, 0.99)
    five = benchmark_forecasts(port, 5, 0.99)
    assert five["gaussian_sqrt"][0] > one["gaussian_sqrt"][0]
    assert five["historical"][0] > one["historical"][0]


def test_measured_risk_table_is_repeatable(window):
    a = measured_risk_table(window, [1, 5], 0.99)
    b = measured_risk_table(window, [1, 5], 0.99)
    pd.testing.assert_frame_equal(a, b)

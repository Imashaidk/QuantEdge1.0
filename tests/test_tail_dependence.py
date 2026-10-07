"""Tests for the horizon tail dependence analysis.

Author: Sameera Ekanayaka
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from src.copulas import StudentTCopula
from src.tail_dependence import (
    block_bootstrap_indices,
    coexceedance,
    gaussian_coexceedance,
    horizon_views,
    pair_tail_table,
    to_ranks,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(1)
    cov = [[1.0, 0.6, -0.3], [0.6, 1.0, -0.2], [-0.3, -0.2, 1.0]]
    x = rng.multivariate_normal(np.zeros(3), cov, size=600) * 0.01
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    return pd.DataFrame(x, index=idx, columns=["A", "B", "C"])


def test_views_are_low_pass_parts_of_the_same_series(returns):
    views = horizon_views(returns, level=4)
    assert list(views) == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(views[0], returns)
    # Each coarser view keeps less variance than the one before it.
    variances = [views[j]["A"].var() for j in views]
    assert all(a > b for a, b in zip(variances, variances[1:]))


def test_coexceedance_limits():
    u = np.linspace(0.001, 0.999, 1000)
    assert coexceedance(u, u, q=0.05) == pytest.approx(1.0, abs=0.01)
    assert coexceedance(u, 1 - u, q=0.05) == 0.0


def test_gaussian_benchmark_matches_independence_at_zero_correlation():
    assert gaussian_coexceedance(0.0, q=0.05) == pytest.approx(0.05, abs=1e-6)
    assert gaussian_coexceedance(0.9, q=0.05) > gaussian_coexceedance(0.3, q=0.05)


def test_ranks_are_in_unit_interval(returns):
    u = to_ranks(returns.to_numpy())
    assert u.min() > 0 and u.max() < 1


def test_bootstrap_is_seeded():
    a = block_bootstrap_indices(500, 50, 10, seed=3)
    b = block_bootstrap_indices(500, 50, 10, seed=3)
    assert a.shape == (10, 500)
    np.testing.assert_array_equal(a, b)


def test_pair_table_change_is_zero_for_daily(returns):
    views = horizon_views(returns, level=3)
    table = pair_tail_table(views, [("A", "B")], reps=20)
    daily = table[table["view"] == 0].iloc[0]
    assert daily["change_vs_daily"] == 0.0
    assert daily["ci_low"] <= daily["lambda_L"] <= daily["ci_high"]


def test_t_copula_tail_dependence_is_averaged_per_pair():
    # One strongly positive pair and one strongly negative pair. The old formula
    # plugged in the average correlation and got almost nothing.
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.8, -0.8], [0.8, 1.0, -0.8], [-0.8, -0.8, 1.0]])
    cov = cov + np.eye(3) * 0.3
    x = rng.multivariate_normal(np.zeros(3), cov, size=3000)
    u = to_ranks(x)
    cop = StudentTCopula().fit(u)
    rho = cop.R[np.triu_indices(3, k=1)]
    arg = -np.sqrt((cop.nu + 1) * (1 - rho) / (1 + rho))
    expected = np.mean(2 * student_t.cdf(arg, df=cop.nu + 1))
    assert cop.lambda_L == pytest.approx(expected)

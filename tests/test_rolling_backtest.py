"""Tests for the rolling backtest.

Author: Sameera Ekanayaka
"""

import numpy as np
import pandas as pd
import pytest

from src.rolling_backtest import (
    compare_to_daily_copula,
    evaluate_forecasts,
    newey_west_tstat,
    run_rolling_forecasts,
)


@pytest.fixture(scope="module")
def forecasts():
    rng = np.random.default_rng(11)
    cov = np.eye(5) * 0.6 + 0.4
    x = rng.multivariate_normal(np.zeros(5), cov, size=420) * 0.01
    idx = pd.date_range("2019-01-01", periods=420, freq="B")
    returns = pd.DataFrame(x, index=idx, columns=["SPY", "QQQ", "TLT", "GLD", "HYG"])
    return run_rolling_forecasts(returns, horizons=[1, 5], window=300, refit_every=60, verbose=False)


def test_forecasts_only_use_past_data(forecasts):
    # The first forecast is made on the last day of the first window.
    assert forecasts["date"].min() == pd.Timestamp("2019-01-01") + pd.offsets.BDay(299)
    # Every forecast has a realised loss strictly after its date.
    assert forecasts.groupby("h")["date"].max().loc[5] < forecasts.groupby("h")["date"].max().loc[1]


def test_evaluation_uses_non_overlapping_windows(forecasts):
    ev = evaluate_forecasts(forecasts)
    n1 = ev[(ev["horizon"] == 1)]["obs"].iloc[0]
    n5 = ev[(ev["horizon"] == 5)]["obs"].iloc[0]
    assert n5 == pytest.approx(n1 / 5, abs=2)


def test_dm_is_zero_for_identical_one_day_models(forecasts):
    dm = compare_to_daily_copula(forecasts)
    row = dm[(dm["horizon"] == 1) & (dm["model"] == "horizon_copula")].iloc[0]
    assert row["mean_fz_diff"] == 0.0


def test_newey_west_matches_plain_t_without_lags():
    d = np.random.default_rng(0).normal(0.1, 1.0, 2000)
    plain = d.mean() / (d.std(ddof=0) / np.sqrt(len(d)))
    assert newey_west_tstat(d, 0) == pytest.approx(plain)

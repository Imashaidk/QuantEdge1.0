"""Tests that every figure and table used in the report can be written.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
import pytest

from src.rolling_backtest import ALL_MODELS, capital_gap, compare_fz, evaluate_forecasts
from src.visualizer import (
    export_rolling_backtest_table,
    export_tail_table,
    export_variance_table,
    plot_fig1_wavelet_mra,
    plot_fig2_tail_by_horizon,
    plot_fig3_capital_gap,
    plot_fig4_rolling_var,
)

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "HYG"]


def _tail_tables():
    rows = []
    for pair in ["Risky-Hedge", "SPY-TLT", "SPY-HYG", "TLT-HYG", "SPY-GLD"]:
        for j in range(6):
            rows.append({"view": j, "horizon": "Daily" if j == 0 else f"> {2 ** (j + 1)}d", "pair": pair,
                         "lambda_L": 0.1 + 0.02 * j, "ci_low": 0.05, "ci_high": 0.3, "change_vs_daily": 0.02 * j,
                         "change_ci_low": -0.004, "change_ci_high": 0.2, "lambda_U": 0.08, "gauss": 0.06})
    table = pd.DataFrame(rows)
    return table[table["pair"] == "Risky-Hedge"], table


@pytest.fixture(scope="module")
def forecasts():
    """Synthetic forecasts with the same columns run_rolling_forecasts returns."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rows = []
    for h in [1, 5, 20, 60]:
        loss = rng.normal(0, 0.01 * np.sqrt(h), len(dates))
        for k, name in enumerate(ALL_MODELS):
            var = 0.025 * np.sqrt(h) * (1 + 0.02 * k) * np.ones(len(dates))
            rows.append(pd.DataFrame({"date": dates, "h": h, "model": name, "VaR": var, "ES": var * 1.2, "loss": loss}))
    return pd.concat(rows, ignore_index=True)


def test_figure_1(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    raw = pd.DataFrame(rng.normal(0, 0.01, (100, 5)), index=dates, columns=TICKERS)
    scales = {s: raw * 0.3 for s in ["D1", "D2", "D3", "D4", "D5", "S5"]}
    out = tmp_path / "fig1.png"
    plot_fig1_wavelet_mra(scales, raw, out_path=out, dpi=80)
    assert out.stat().st_size > 1000


def test_figure_2(tmp_path):
    sleeves, pairs = _tail_tables()
    out = tmp_path / "fig2.png"
    plot_fig2_tail_by_horizon(sleeves, pairs, out_path=out, dpi=80)
    assert out.stat().st_size > 1000


def test_figures_3_and_4(tmp_path, forecasts):
    plot_fig3_capital_gap(capital_gap(forecasts), out_path=tmp_path / "fig3.png", dpi=80)
    plot_fig4_rolling_var(forecasts, horizon=20, out_path=tmp_path / "fig4.png", dpi=80)
    assert (tmp_path / "fig3.png").stat().st_size > 1000
    assert (tmp_path / "fig4.png").stat().st_size > 1000


def test_tail_table_has_no_negative_zero(tmp_path):
    sleeves, pairs = _tail_tables()
    export_tail_table(sleeves, pairs, out_dir=tmp_path)
    text = (tmp_path / "tail_by_horizon.tex").read_text()
    assert "\\begin{table}" in text
    assert "-0.00" not in text


def test_backtest_table_has_one_copula_row_at_one_day(tmp_path, forecasts):
    tested = forecasts[forecasts["h"].isin([1, 5, 20])]
    export_rolling_backtest_table(evaluate_forecasts(tested), compare_fz(tested), out_dir=tmp_path)
    one_day = [l for l in (tmp_path / "backtest_metrics.tex").read_text().splitlines() if l.startswith("1d &")]
    assert len(one_day) == 3


def test_variance_table(tmp_path):
    shares = pd.DataFrame({"SPY": [60.0, 40.0]}, index=["D1", "S5"])
    export_variance_table(shares, out_dir=tmp_path)
    assert "\\centering" in (tmp_path / "variance_decomposition.tex").read_text()

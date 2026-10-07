"""Runs the whole analysis and regenerates every number, table and figure in the report.

Question: does tail dependence change with the investment horizon, and what does
ignoring this do to a portfolio's measured risk?

Steps:
  1. Load daily prices and compute log returns
  2. MODWT multiresolution analysis and variance by scale
  3. Tail co-exceedance by horizon view, with block bootstrap intervals
  4. Rolling out-of-sample backtest of the VaR models
  5. Measured risk by model on the latest window
  6. Figures and LaTeX tables
  7. Summary

Usage:
  python run_all.py

Author: Sameera Ekanayaka
"""

import sys
import time
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import pandas as pd

from src.config import (
    ALPHA_VAR_99,
    BACKTEST_HORIZONS,
    DEFAULT_PORTFOLIO_WEIGHTS,
    FIGURES_DIR,
    REFIT_EVERY,
    RESULTS_DIR,
    ROLLING_WINDOW,
    TABLES_DIR,
    TICKERS,
    WAVELET_FAMILY,
    WAVELET_LEVEL,
)
from src.data_loader import load_returns
from src.horizon_var import MODEL_LABELS, measured_risk_table
from src.rolling_backtest import (
    capital_gap,
    compare_to_daily_copula,
    evaluate_forecasts,
    run_rolling_forecasts,
    stress_breaches,
)
from src.tail_dependence import run_tail_dependence_analysis
from src.visualizer import (
    export_rolling_backtest_table,
    export_tail_table,
    export_variance_table,
    plot_fig1_wavelet_mra,
    plot_fig2_tail_by_horizon,
    plot_fig3_rolling_var,
    plot_fig4_capital_gap,
)
from src.wavelets import compute_scale_variance_decomposition, decompose_multiscale, verify_additivity

# Horizons for the capital comparison. 60 days is too few non-overlapping windows
# for coverage tests, so it only appears in the capital comparison.
CAPITAL_HORIZONS = sorted(set(BACKTEST_HORIZONS) | {60})

STRESS_PERIODS = {
    "2008 crisis": ("2008-09-01", "2009-03-31"),
    "COVID 2020": ("2020-02-15", "2020-04-30"),
    "2022 rates shock": ("2022-01-01", "2022-10-31"),
}


def step(title: str) -> float:
    print(f"\n{title}")
    return time.time()


def main() -> None:
    t_total = time.time()
    pd.set_option("display.width", 160)

    t0 = step("[1/7] Loading data")
    returns = load_returns()
    print(f"  Assets: {list(returns.columns)}")
    print(f"  {returns.index[0].date()} to {returns.index[-1].date()}, {len(returns)} trading days")
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = step(f"[2/7] MODWT multiresolution analysis ({WAVELET_FAMILY}, level {WAVELET_LEVEL})")
    decomposed = decompose_multiscale(returns, wavelet=WAVELET_FAMILY, level=WAVELET_LEVEL)
    assert verify_additivity(returns, decomposed, tol=1e-10), "MRA does not add back up to the returns"
    var_share = compute_scale_variance_decomposition(decomposed, normalize=True)
    print("  Share of variance by scale (%):")
    print(var_share.round(1).to_string())
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = step("[3/7] Tail co-exceedance by horizon view")
    tail = run_tail_dependence_analysis(returns, weights=DEFAULT_PORTFOLIO_WEIGHTS)
    for name in ["sleeves", "pairs", "copulas"]:
        tail[name].to_csv(RESULTS_DIR / f"tail_{name}.csv", index=False, float_format="%.6f")
    cols = ["horizon", "lambda_L", "ci_low", "ci_high", "change_vs_daily", "change_ci_low", "change_ci_high", "lambda_U", "gauss"]
    print("  Risky sleeve vs hedge sleeve:")
    print(tail["sleeves"][cols].round(3).to_string(index=False))
    print("  Lower tail by pair:")
    print(tail["pairs"].pivot(index="pair", columns="view", values="lambda_L").round(2).to_string())
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = step(f"[4/7] Rolling backtest (window {ROLLING_WINDOW} days, refit every {REFIT_EVERY} days)")
    forecasts = run_rolling_forecasts(returns, horizons=CAPITAL_HORIZONS, alpha=ALPHA_VAR_99)
    forecasts.to_csv(RESULTS_DIR / "rolling_forecasts.csv", index=False, float_format="%.6f")
    tested = forecasts[forecasts["h"].isin(BACKTEST_HORIZONS)]
    evaluation = evaluate_forecasts(tested, alpha=ALPHA_VAR_99)
    dm = compare_to_daily_copula(tested, alpha=ALPHA_VAR_99)
    gap = capital_gap(forecasts)
    stress = stress_breaches(tested, STRESS_PERIODS)
    for name, df in [("backtest_evaluation", evaluation), ("backtest_dm", dm), ("capital_gap", gap), ("stress_breaches", stress)]:
        df.to_csv(RESULTS_DIR / f"{name}.csv", index=False, float_format="%.6f")
    print(f"  Forecast dates: {forecasts['date'].min().date()} to {forecasts['date'].max().date()}")
    print(evaluation.round(4).to_string(index=False))
    print("  FZ score against the daily copula (negative = better):")
    print(dm.round(4).to_string(index=False))
    print("  VaR change against the daily copula:")
    print(gap.round(4).to_string(index=False))
    if not stress.empty:
        print("  Breaches in stress periods:")
        print(stress.pivot_table(index=["period", "horizon"], columns="model", values="breaches").to_string())
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = step("[5/7] Measured risk on the latest window")
    latest = returns.iloc[-ROLLING_WINDOW:]
    risk_latest = measured_risk_table(latest, CAPITAL_HORIZONS, ALPHA_VAR_99, weights=DEFAULT_PORTFOLIO_WEIGHTS)
    risk_latest.to_csv(RESULTS_DIR / "measured_risk_latest.csv", index=False, float_format="%.6f")
    print(f"  Window {latest.index[0].date()} to {latest.index[-1].date()}")
    print(risk_latest.round(4).to_string(index=False))
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = step("[6/7] Figures and tables")
    plot_fig1_wavelet_mra(decomposed, returns, primary_asset="SPY", secondary_asset="TLT",
                          out_path=FIGURES_DIR / "fig1_wavelet_mra_decomposition.png")
    plot_fig2_tail_by_horizon(tail["sleeves"], tail["pairs"], out_path=FIGURES_DIR / "fig2_tail_dependence_vs_horizon.png")
    plot_fig3_rolling_var(tested, horizon=20, out_path=FIGURES_DIR / "fig3_backtest_var_exceedances.png")
    plot_fig4_capital_gap(gap, out_path=FIGURES_DIR / "fig4_capital_gap_by_horizon.png")
    export_variance_table(var_share, out_dir=TABLES_DIR)
    export_tail_table(tail["sleeves"], tail["pairs"], out_dir=TABLES_DIR)
    export_rolling_backtest_table(evaluation, dm, out_dir=TABLES_DIR)
    print(f"  done in {time.time() - t0:.1f}s")

    step("[7/7] Summary")
    sl = tail["sleeves"].set_index("view")
    last = sl.index.max()
    print(f"  Risky vs hedge lower-tail co-exceedance: daily {sl.loc[0, 'lambda_L']:.2f}, "
          f"{sl.loc[last, 'horizon']} {sl.loc[last, 'lambda_L']:.2f} "
          f"(change {sl.loc[last, 'change_vs_daily']:+.2f}, 90% interval "
          f"{sl.loc[last, 'change_ci_low']:+.2f} to {sl.loc[last, 'change_ci_high']:+.2f})")
    hc = gap[gap["model"] == "horizon_copula"].set_index("horizon")
    for h in CAPITAL_HORIZONS:
        if h == 1:
            continue
        print(f"  {h:>2}d VaR, horizon copula vs daily copula: {hc.loc[h, 'var_ratio_mean']:+.1%} on average "
              f"(10th to 90th percentile {hc.loc[h, 'var_ratio_p10']:+.1%} to {hc.loc[h, 'var_ratio_p90']:+.1%})")
    best = evaluation.loc[evaluation.groupby("horizon")["fz"].idxmin(), ["horizon", "model"]]
    for _, r in best.iterrows():
        print(f"  Best FZ score at {r['horizon']}d: {MODEL_LABELS[r['model']]}")

    print(f"\nFinished in {time.time() - t_total:.0f} seconds.")


if __name__ == "__main__":
    main()

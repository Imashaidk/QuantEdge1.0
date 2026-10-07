"""Rolling out-of-sample backtest.

At the close of every trading day t we forecast the VaR and ES of the portfolio
loss over days t+1..t+h, using only data up to t:

  - every REFIT_EVERY days all models are re-estimated on the last ROLLING_WINDOW days
  - in between, the GARCH volatilities are rolled forward each day with the fixed
    parameters, so the forecasts still react to new data

Coverage tests (Kupiec, Christoffersen) need independent observations, so for h > 1
they use every h-th forecast, which makes the loss windows non-overlapping.
The FZ score is a plain average and uses every forecast.

We also compare the proposed horizon copula against the daily copula and the
sqrt(h) rule with a Diebold-Mariano test on the FZ score, using Newey-West
standard errors because overlapping h-day losses are autocorrelated.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.backtest import (
    christoffersen_independence_test,
    classify_basel_traffic_light,
    kupiec_pof_test,
)
from src.config import (
    ALPHA_VAR_99,
    BACKTEST_HORIZONS,
    DEFAULT_PORTFOLIO_WEIGHTS,
    REFIT_EVERY,
    ROLLING_WINDOW,
)
from src.horizon_var import BENCHMARK_MODELS, COPULA_MODELS, HorizonVaRModel, benchmark_forecasts

ALL_MODELS: List[str] = BENCHMARK_MODELS + COPULA_MODELS


def run_rolling_forecasts(
    returns: pd.DataFrame,
    horizons: Sequence[int] = BACKTEST_HORIZONS,
    alpha: float = ALPHA_VAR_99,
    weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
    window: int = ROLLING_WINDOW,
    refit_every: int = REFIT_EVERY,
    verbose: bool = True,
) -> pd.DataFrame:
    """Daily VaR/ES forecasts for every model and horizon, with the realised loss."""
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    x = returns.to_numpy(dtype=float)
    port = x @ w
    dates = returns.index
    n = len(returns)

    rows = []
    model = None
    first = window - 1
    for t in range(first, n - 1):
        if (t - first) % refit_every == 0:
            model = HorizonVaRModel.fit(returns.iloc[t - window + 1 : t + 1], horizons, weights=w)
            if verbose and ((t - first) // refit_every) % 12 == 0:
                print(f"  refit at {dates[t].date()}")
        else:
            model.update(x[t])

        port_window = port[t - window + 1 : t + 1]
        for h in horizons:
            if t + h >= n:
                continue
            loss = -float(port[t + 1 : t + 1 + h].sum())
            fc = {**benchmark_forecasts(port_window, h, alpha), **model.forecast(h, alpha)}
            for name in ALL_MODELS:
                var, es = fc[name]
                rows.append({"date": dates[t], "h": h, "model": name, "VaR": var, "ES": es, "loss": loss})

    return pd.DataFrame(rows)


def newey_west_tstat(d: np.ndarray, lags: int) -> float:
    """t statistic of mean(d) with Newey-West (Bartlett) standard errors."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    e = d - d.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] @ e[:-k]) / n
    return float(d.mean() / np.sqrt(max(s, 1e-18) / n))


def fz_scores(f: pd.DataFrame, alpha: float) -> np.ndarray:
    """Per-forecast FZ0 score (lower is better). Same formula as backtest.fissler_ziegel_loss."""
    y = f["loss"].to_numpy(dtype=float)
    v = np.maximum(f["VaR"].to_numpy(dtype=float), 1e-6)
    e = np.maximum(f["ES"].to_numpy(dtype=float), v + 1e-6)
    hit = (y > v).astype(float)
    return (v + (y - v) * hit / (1.0 - alpha)) / e + np.log(e) - 1.0


def evaluate_forecasts(forecasts: pd.DataFrame, alpha: float = ALPHA_VAR_99) -> pd.DataFrame:
    """Coverage tests, FZ score and average capital for each model and horizon."""
    rows = []
    for h, fh in forecasts.groupby("h"):
        for name in ALL_MODELS:
            f = fh[fh["model"] == name].sort_values("date").reset_index(drop=True)
            nonoverlap = f.iloc[::h]
            hits = (nonoverlap["loss"] > nonoverlap["VaR"]).astype(int).to_numpy()
            x, n_obs = int(hits.sum()), len(hits)
            _, p_kupiec, _ = kupiec_pof_test(x, n_obs, alpha=alpha)
            _, p_ind, _ = christoffersen_independence_test(hits)
            row = {
                "horizon": int(h),
                "model": name,
                "obs": n_obs,
                "breaches": x,
                "expected": round(n_obs * (1 - alpha), 1),
                "breach_rate": x / n_obs,
                "kupiec_p": p_kupiec,
                "christoffersen_p": p_ind,
                "fz": float(fz_scores(f, alpha).mean()),
                "avg_var": float(f["VaR"].mean()),
                "all_breaches": int((f["loss"] > f["VaR"]).sum()),
            }
            if h == 1:
                row["basel_zone"] = basel_worst_zone(f, alpha)
            rows.append(row)
    return pd.DataFrame(rows)


def basel_worst_zone(f: pd.DataFrame, alpha: float) -> str:
    """Worst Basel traffic light zone over any rolling 250-day window."""
    hits = (f["loss"] > f["VaR"]).astype(int).rolling(250).sum().dropna()
    if hits.empty:
        return "n/a"
    worst = int(hits.max())
    return classify_basel_traffic_light(worst, 250, alpha=alpha)["zone"]


def compare_fz(forecasts: pd.DataFrame, alpha: float = ALPHA_VAR_99, base: str = "daily_copula") -> pd.DataFrame:
    """Diebold-Mariano test of every model's FZ score against a base model.

    A negative mean difference means the model scores better than the base.
    """
    rows = []
    for h, fh in forecasts.groupby("h"):
        s_base = fz_scores(fh[fh["model"] == base].sort_values("date"), alpha)
        for name in ALL_MODELS:
            if name == base:
                continue
            other = fh[fh["model"] == name].sort_values("date")
            d = fz_scores(other, alpha) - s_base
            lags = max(int(h), int(np.floor(4 * (len(d) / 100) ** (2 / 9))))
            t = newey_west_tstat(d, lags)
            rows.append({
                "horizon": int(h),
                "base": base,
                "model": name,
                "mean_fz_diff": float(d.mean()),
                "t_stat": t,
                "p_value": float(2 * norm.sf(abs(t))),
            })
    return pd.DataFrame(rows)


def stress_breaches(forecasts: pd.DataFrame, periods: Dict[str, tuple]) -> pd.DataFrame:
    """Breach counts (all forecasts) inside named stress periods."""
    rows = []
    for label, (start, end) in periods.items():
        sub = forecasts[(forecasts["date"] >= start) & (forecasts["date"] <= end)]
        if sub.empty:
            continue
        for (h, name), f in sub.groupby(["h", "model"]):
            rows.append({
                "period": label,
                "horizon": int(h),
                "model": name,
                "days": len(f),
                "breaches": int((f["loss"] > f["VaR"]).sum()),
            })
    return pd.DataFrame(rows)


def capital_gap(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Average VaR and ES of each copula model relative to the daily copula."""
    rows = []
    for h, fh in forecasts.groupby("h"):
        base = fh[fh["model"] == "daily_copula"].set_index("date")
        for name in ALL_MODELS:
            m = fh[fh["model"] == name].set_index("date")
            rows.append({
                "horizon": int(h),
                "model": name,
                "var_ratio_mean": float((m["VaR"] / base["VaR"]).mean() - 1.0),
                "var_ratio_p10": float((m["VaR"] / base["VaR"]).quantile(0.10) - 1.0),
                "var_ratio_p90": float((m["VaR"] / base["VaR"]).quantile(0.90) - 1.0),
                "es_ratio_mean": float((m["ES"] / base["ES"]).mean() - 1.0),
            })
    return pd.DataFrame(rows)

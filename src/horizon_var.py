"""Horizon-aware copula VaR and the models we compare it with.

All copula models share the same marginal model, so any difference between them
comes only from the dependence structure:

  margins : GJR-GARCH(1,1) with t errors for each asset, EVT tails on the
            standardised residuals. The h-day volatility is the sum of the GARCH
            variance forecasts for days 1..h, so volatility mean reversion is
            included and is not just sqrt(h) times today's volatility.

  "daily_sqrt"     : 1-day VaR from the daily copula, scaled by sqrt(h).
                     This is the usual practice.
  "daily_copula"   : h-day margins, but the copula is fitted to daily data.
                     This ignores any change in dependence with the horizon.
  "horizon_copula" : h-day margins and a copula fitted to the wavelet horizon
                     view that matches h (see tail_dependence.py).
                     This is the proposed model.

Comparing daily_copula with horizon_copula isolates what ignoring the horizon
dependence does to measured risk. Comparing daily_sqrt with daily_copula shows
the separate effect of the sqrt(h) rule.

The two simple benchmarks, historical simulation and a Gaussian sqrt(h) rule,
are computed on the portfolio return series directly.

Author: Sameera Ekanayaka
"""

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

from src.config import (
    DEFAULT_PORTFOLIO_WEIGHTS,
    EVT_TAIL_PERCENTILE,
    HORIZON_VIEW,
    RANDOM_SEED,
    VAR_SIMULATIONS,
)
from src.copulas import StudentTCopula
from src.margins import GARCH_EVT_Margin
from src.tail_dependence import horizon_views, to_ranks

COPULA_MODELS: List[str] = ["daily_sqrt", "daily_copula", "horizon_copula"]
BENCHMARK_MODELS: List[str] = ["historical", "gaussian_sqrt"]

MODEL_LABELS: Dict[str, str] = {
    "historical": "Historical simulation",
    "gaussian_sqrt": "Gaussian, sqrt(h)",
    "daily_sqrt": "Daily copula, sqrt(h)",
    "daily_copula": "Daily copula, h-day vol",
    "horizon_copula": "Horizon copula (proposed)",
}


@dataclass
class AssetGarch:
    """GJR-GARCH(1,1) parameters for one asset, in percent units."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    last_sigma2: float
    last_eps: float

    def update(self, r_pct: float) -> None:
        """Rolls the variance forward one day after observing return r_pct.
        
        sigma^2(t+1) = omega + (alpha + gamma * I[eps < 0]) * eps^2 + beta * sigma^2(t)
        using the newly observed residual eps = r_pct - mu.
        """
        eps = r_pct - self.mu
        neg = 1.0 if eps < 0.0 else 0.0
        self.last_sigma2 = self.omega + (self.alpha + self.gamma * neg) * (eps ** 2) + self.beta * self.last_sigma2
        self.last_eps = eps

    def horizon_variance(self, h: int) -> float:
        """Sum of the expected daily variances over the next h days.
        
        last_sigma2 holds the 1-step ahead conditional variance for day 1.
        Future days mean-revert according to GJR persistence.
        """
        persistence = self.alpha + 0.5 * self.gamma + self.beta
        s2 = self.last_sigma2
        total = 0.0
        for _ in range(h):
            total += s2
            s2 = self.omega + persistence * s2
        return total


def fit_asset_garch(r: np.ndarray) -> Tuple[AssetGarch, np.ndarray]:
    """Fits GJR-GARCH(1,1)-t to daily returns. Returns parameters and standardised residuals."""
    r_pct = np.asarray(r, dtype=float) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    p = res.params
    sigma = np.asarray(res.conditional_volatility, dtype=float)
    resid = np.asarray(res.resid, dtype=float)
    
    # 1-step ahead conditional variance for the day following the sample
    neg = 1.0 if resid[-1] < 0.0 else 0.0
    s2_next = float(
        p["omega"]
        + (p["alpha[1]"] + p["gamma[1]"] * neg) * (resid[-1] ** 2)
        + p["beta[1]"] * (sigma[-1] ** 2)
    )
    garch = AssetGarch(
        mu=float(p["mu"]),
        omega=float(p["omega"]),
        alpha=float(p["alpha[1]"]),
        gamma=float(p["gamma[1]"]),
        beta=float(p["beta[1]"]),
        last_sigma2=s2_next,
        last_eps=float(resid[-1]),
    )
    return garch, resid / sigma


@dataclass
class HorizonVaRModel:
    """Everything fitted on one estimation window."""

    weights: np.ndarray
    horizons: Sequence[int]
    garch: List[AssetGarch] = field(default_factory=list)
    daily_draws: np.ndarray = None
    horizon_draws: Dict[int, np.ndarray] = field(default_factory=dict)
    copula_nu: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        window: pd.DataFrame,
        horizons: Sequence[int],
        weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
        n_sims: int = VAR_SIMULATIONS,
        seed: int = RANDOM_SEED,
    ) -> "HorizonVaRModel":
        model = cls(weights=np.asarray(weights, dtype=float) / np.sum(weights), horizons=list(horizons))

        margins = []
        for col in window.columns:
            g, z = fit_asset_garch(window[col].to_numpy())
            model.garch.append(g)
            margins.append(GARCH_EVT_Margin(asset_name=col, tail_percentile=EVT_TAIL_PERCENTILE).fit_residuals(z))

        views = horizon_views(window)
        needed = {0} | {HORIZON_VIEW[h] for h in horizons}
        for j in sorted(needed):
            cop = StudentTCopula().fit(to_ranks(views[j].to_numpy(dtype=float)))
            u = cop.sample(n_samples=n_sims, seed=seed)
            # Common random numbers: the same seed for every view, so differences
            # between models are not simulation noise.
            z = np.column_stack([margins[k].ppf(u[:, k]) for k in range(len(margins))])
            model.copula_nu[j] = cop.nu
            if j == 0:
                model.daily_draws = z
            for h in horizons:
                if HORIZON_VIEW[h] == j:
                    model.horizon_draws[h] = z
        return model

    def update(self, daily_returns: np.ndarray) -> None:
        """Feeds one day of observed returns into every asset's GARCH state."""
        for g, r in zip(self.garch, daily_returns):
            g.update(float(r) * 100.0)

    def _var_es(self, draws: np.ndarray, scale: np.ndarray, drift: float, alpha: float) -> Tuple[float, float]:
        losses = -(drift + draws @ (self.weights * scale))
        var = float(np.quantile(losses, alpha))
        es = float(losses[losses >= var].mean())
        return var, es

    def forecast(self, h: int, alpha: float) -> Dict[str, Tuple[float, float]]:
        """VaR and ES of the h-day portfolio loss for the three copula models."""
        mu = np.array([g.mu for g in self.garch]) / 100.0
        sd1 = np.sqrt([g.horizon_variance(1) for g in self.garch]) / 100.0
        sdh = np.sqrt([g.horizon_variance(h) for g in self.garch]) / 100.0
        drift1 = float(self.weights @ mu)

        v1, e1 = self._var_es(self.daily_draws, sd1, drift1, alpha)
        out = {"daily_sqrt": (v1 * np.sqrt(h), e1 * np.sqrt(h))}
        out["daily_copula"] = self._var_es(self.daily_draws, sdh, drift1 * h, alpha)
        out["horizon_copula"] = self._var_es(self.horizon_draws[h], sdh, drift1 * h, alpha)
        return out


def benchmark_forecasts(port_window: np.ndarray, h: int, alpha: float) -> Dict[str, Tuple[float, float]]:
    """Historical simulation on overlapping h-day sums and a Gaussian sqrt(h) rule."""
    r = np.asarray(port_window, dtype=float)
    hsum = np.convolve(r, np.ones(h), mode="valid") if h > 1 else r
    losses = -hsum
    var_hs = float(np.quantile(losses, alpha))
    es_hs = float(losses[losses >= var_hs].mean())

    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = norm.ppf(alpha)
    var_g = -mu * h + z * sd * np.sqrt(h)
    es_g = -mu * h + sd * np.sqrt(h) * norm.pdf(z) / (1.0 - alpha)
    return {"historical": (var_hs, es_hs), "gaussian_sqrt": (float(var_g), float(es_g))}


def measured_risk_table(
    window: pd.DataFrame,
    horizons: Sequence[int],
    alpha: float,
    weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
) -> pd.DataFrame:
    """VaR and ES by model and horizon for one estimation window.

    Answers the second half of the question for that date: how much does the
    measured risk change if you use the dependence that matches the horizon?
    """
    model = HorizonVaRModel.fit(window, horizons, weights=weights)
    port = window.to_numpy() @ (np.asarray(weights) / np.sum(weights))
    rows = []
    for h in horizons:
        fc = {**benchmark_forecasts(port, h, alpha), **model.forecast(h, alpha)}
        base_var, base_es = fc["daily_copula"]
        for name in BENCHMARK_MODELS + COPULA_MODELS:
            var, es = fc[name]
            rows.append({
                "horizon": h,
                "model": name,
                "VaR": var,
                "ES": es,
                "VaR_vs_daily_copula": var / base_var - 1.0,
                "ES_vs_daily_copula": es / base_es - 1.0,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.config import ALPHA_VAR_99, ROLLING_WINDOW
    from src.data_loader import load_returns

    returns = load_returns()
    window = returns.iloc[-ROLLING_WINDOW:]
    table = measured_risk_table(window, [1, 5, 20, 60], ALPHA_VAR_99)
    print(f"Window {window.index[0].date()} to {window.index[-1].date()}")
    print(table.round(4).to_string(index=False))

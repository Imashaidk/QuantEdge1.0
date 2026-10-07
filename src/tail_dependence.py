"""Tail dependence across investment horizons.

The question we want to answer is whether assets crash together more (or less)
when you look at longer holding periods. We measure this on wavelet "horizon
views" of the returns:

    view j = 0 : raw daily returns
    view j >= 1: the low-pass part of the MODWT MRA, sum of D_{j+1}..D_J plus S_J,
                 which keeps only moves that last longer than 2^(j+1) days

An h-day return is itself a low-pass filter of daily returns, so view j is a
smooth stand-in for holding the portfolio about 2^(j+1) days. The advantage
over summing returns into non-overlapping h-day blocks is that we keep every
daily observation instead of throwing away h-1 out of every h.

Tail co-exceedance at level q is P(U1 <= q, U2 <= q) / q on rank data. It is 0.05
for independent assets at q = 0.05 and 1 for perfectly dependent ones. We compare
it with what a Gaussian copula with the same correlation would give, and put
moving block bootstrap intervals around it, because the coarse views have far
fewer independent observations than their length suggests.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, rankdata

from src.config import (
    DEFAULT_PORTFOLIO_WEIGHTS,
    RANDOM_SEED,
    TAIL_QUANTILE,
    TAIL_BOOTSTRAP_REPS,
    WAVELET_BOUNDARY,
    WAVELET_FAMILY,
    WAVELET_LEVEL,
)
from src.wavelets import mra_decompose

# Risky sleeve vs hedge sleeve of the portfolio. Used for the single headline
# question: does the hedge still work in a crash at longer horizons?
RISKY_SLEEVE: List[str] = ["SPY", "QQQ", "HYG"]
HEDGE_SLEEVE: List[str] = ["TLT", "GLD"]

KEY_PAIRS: List[Tuple[str, str]] = [
    ("SPY", "QQQ"),
    ("SPY", "HYG"),
    ("SPY", "TLT"),
    ("SPY", "GLD"),
    ("TLT", "HYG"),
    ("TLT", "GLD"),
]


def view_label(j: int) -> str:
    """Readable name for horizon view j."""
    if j == 0:
        return "Daily"
    return f"> {2 ** (j + 1)}d"


def horizon_views(
    df_returns: pd.DataFrame,
    level: int = WAVELET_LEVEL,
    wavelet: str = WAVELET_FAMILY,
    boundary: str = WAVELET_BOUNDARY,
) -> Dict[int, pd.DataFrame]:
    """Builds the raw series (j=0) and the low-pass views j = 1..level."""
    mra = {
        col: mra_decompose(df_returns[col].to_numpy(dtype=float), wavelet=wavelet, level=level, boundary=boundary)
        for col in df_returns.columns
    }
    views: Dict[int, pd.DataFrame] = {0: df_returns.copy()}
    for j in range(1, level + 1):
        data = {}
        for col in df_returns.columns:
            parts = [mra[col][f"D{k}"] for k in range(j + 1, level + 1)]
            data[col] = np.sum(parts, axis=0) + mra[col][f"S{level}"] if parts else mra[col][f"S{level}"]
        views[j] = pd.DataFrame(data, index=df_returns.index, columns=df_returns.columns)
    return views


def to_ranks(x: np.ndarray) -> np.ndarray:
    """Rank transform each column of x to (0, 1)."""
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T
    n = x.shape[0]
    return np.column_stack([rankdata(x[:, k]) / (n + 1.0) for k in range(x.shape[1])])


def coexceedance(u1: np.ndarray, u2: np.ndarray, q: float = TAIL_QUANTILE, lower: bool = True) -> float:
    """Empirical tail co-exceedance P(both in the tail) / q on rank data."""
    if lower:
        return float(np.mean((u1 <= q) & (u2 <= q)) / q)
    return float(np.mean((u1 >= 1.0 - q) & (u2 >= 1.0 - q)) / q)


def gaussian_coexceedance(rho: float, q: float = TAIL_QUANTILE) -> float:
    """Co-exceedance a Gaussian copula with correlation rho would produce."""
    z = norm.ppf(q)
    rho = float(np.clip(rho, -0.999, 0.999))
    p = multivariate_normal.cdf([z, z], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    return float(p / q)


def normal_scores_corr(u1: np.ndarray, u2: np.ndarray) -> float:
    """Correlation of normal scores, the natural Gaussian copula estimate."""
    return float(np.corrcoef(norm.ppf(u1), norm.ppf(u2))[0, 1])


def block_bootstrap_indices(n: int, block: int, reps: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Moving block bootstrap: glue together random blocks of consecutive days.

    Keeping blocks intact preserves the serial dependence inside each view, which is
    strong for the smooth views. Indices wrap around at the end of the sample.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]


# One block length for every view, so the same resampled days are used across
# horizons and we can put an interval on the change relative to daily.
# About six months: long enough to hold a few full cycles of the slowest view.
BOOTSTRAP_BLOCK: int = 126


def sleeve_returns(df: pd.DataFrame, weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS) -> pd.DataFrame:
    """Weighted return of the risky sleeve and of the hedge sleeve."""
    w = pd.Series(np.asarray(weights, dtype=float), index=df.columns)
    risky = (df[RISKY_SLEEVE] * w[RISKY_SLEEVE]).sum(axis=1) / w[RISKY_SLEEVE].sum()
    hedge = (df[HEDGE_SLEEVE] * w[HEDGE_SLEEVE]).sum(axis=1) / w[HEDGE_SLEEVE].sum()
    return pd.DataFrame({"Risky": risky, "Hedge": hedge}, index=df.index)


def pair_tail_table(
    views: Dict[int, pd.DataFrame],
    pairs: Sequence[Tuple[str, str]],
    q: float = TAIL_QUANTILE,
    reps: int = TAIL_BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Lower-tail co-exceedance per pair and view, with 90% bootstrap intervals.

    Also reports the change against the daily view and an interval for that change,
    computed from the same bootstrap draws.
    """
    n = len(views[0])
    idx = block_bootstrap_indices(n, BOOTSTRAP_BLOCK, reps, seed=seed)
    cols = list(views[0].columns)

    rows = []
    for a, b in pairs:
        ia, ib = cols.index(a), cols.index(b)
        boot_daily = None
        for j in sorted(views):
            x = views[j].to_numpy(dtype=float)[:, [ia, ib]]
            u = to_ranks(x)
            est = coexceedance(u[:, 0], u[:, 1], q)
            rho = normal_scores_corr(u[:, 0], u[:, 1])
            boot = np.empty(reps)
            for r in range(reps):
                ub = to_ranks(x[idx[r]])
                boot[r] = coexceedance(ub[:, 0], ub[:, 1], q)
            if j == 0:
                boot_daily, est_daily = boot, est
            diff = boot - boot_daily
            rows.append({
                "view": j,
                "horizon": view_label(j),
                "pair": f"{a}-{b}",
                "lambda_L": est,
                "ci_low": float(np.percentile(boot, 5)),
                "ci_high": float(np.percentile(boot, 95)),
                "change_vs_daily": est - est_daily,
                "change_ci_low": float(np.percentile(diff, 5)),
                "change_ci_high": float(np.percentile(diff, 95)),
                "lambda_U": coexceedance(u[:, 0], u[:, 1], q, lower=False),
                "gauss": gaussian_coexceedance(rho, q),
                "rho": rho,
            })
    return pd.DataFrame(rows)


def pair_copula_table(
    views: Dict[int, pd.DataFrame],
    pairs: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    """Best bivariate copula family (by BIC) for each pair and view.

    With two assets all five families use the full likelihood, so their BIC
    values can be compared directly.
    """
    from src.copulas import run_scale_copula_tournament

    rows = []
    for j in sorted(views):
        for a, b in pairs:
            u = pd.DataFrame(to_ranks(views[j][[a, b]].to_numpy(dtype=float)), columns=[a, b])
            res = run_scale_copula_tournament(u, scale_name=f"{view_label(j)} {a}-{b}")
            best = res["fitted_copula_obj"]
            rows.append({
                "view": j,
                "horizon": view_label(j),
                "pair": f"{a}-{b}",
                "best_copula": res["best_copula"],
                "theo_lambda_L": float(best.lambda_L),
                "theo_lambda_U": float(best.lambda_U),
                "t_nu": float(res["all_models"]["student_t"].nu),
                "bic_gap_to_gaussian": float(res["bic_scores"]["gaussian"] - res["bic_scores"][res["best_copula"]]),
            })
    return pd.DataFrame(rows)


def run_tail_dependence_analysis(
    df_returns: pd.DataFrame,
    weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    q: float = TAIL_QUANTILE,
    reps: int = TAIL_BOOTSTRAP_REPS,
) -> Dict[str, pd.DataFrame]:
    """Full horizon analysis: key asset pairs plus the risky vs hedge sleeve."""
    pairs = list(pairs) if pairs is not None else KEY_PAIRS
    views = horizon_views(df_returns)
    pair_table = pair_tail_table(views, pairs, q=q, reps=reps)

    sleeve_views = horizon_views(sleeve_returns(df_returns, weights))
    sleeve_table = pair_tail_table(sleeve_views, [("Risky", "Hedge")], q=q, reps=reps)
    copula_table = pair_copula_table(sleeve_views, [("Risky", "Hedge")])
    copula_table = pd.concat([copula_table, pair_copula_table(views, pairs)], ignore_index=True)

    return {"pairs": pair_table, "sleeves": sleeve_table, "copulas": copula_table, "views": views}


def block_length_robustness(
    views: Dict[int, pd.DataFrame],
    pair: Tuple[str, str] = ("SPY", "HYG"),
    target_view: int = 5,
    blocks: Sequence[int] = (63, 126, 252),
    q: float = TAIL_QUANTILE,
    reps: int = TAIL_BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluates change in lower-tail co-exceedance vs daily across different block lengths.

    Supports the block-length robustness check quoted in the report.
    """
    cols = list(views[0].columns)
    ia, ib = cols.index(pair[0]), cols.index(pair[1])
    x_daily = views[0].to_numpy(dtype=float)[:, [ia, ib]]
    x_long = views[target_view].to_numpy(dtype=float)[:, [ia, ib]]
    n = len(views[0])

    u_d = to_ranks(x_daily)
    u_l = to_ranks(x_long)
    est_change = coexceedance(u_l[:, 0], u_l[:, 1], q) - coexceedance(u_d[:, 0], u_d[:, 1], q)

    rows = []
    for b in blocks:
        idx = block_bootstrap_indices(n, b, reps=reps, seed=seed)
        diff = np.empty(reps)
        for r in range(reps):
            ub_d = to_ranks(x_daily[idx[r]])
            ub_l = to_ranks(x_long[idx[r]])
            diff[r] = coexceedance(ub_l[:, 0], ub_l[:, 1], q) - coexceedance(ub_d[:, 0], ub_d[:, 1], q)
        rows.append({
            "pair": f"{pair[0]}-{pair[1]}",
            "block_length": b,
            "change_vs_daily": est_change,
            "ci_low": float(np.percentile(diff, 5)),
            "ci_high": float(np.percentile(diff, 95)),
        })
    return pd.DataFrame(rows)


def spy_tlt_crash_days(
    views: Dict[int, pd.DataFrame],
    pair: Tuple[str, str] = ("SPY", "TLT"),
    target_view: int = 5,
    q: float = TAIL_QUANTILE,
) -> pd.DataFrame:
    """Counts joint crash days (both in lower q tail) for SPY-TLT at the target horizon view.

    Outputs counts by episode/year and total, confirming the 2009 and 2022 concentration.
    """
    v = views[target_view]
    u = to_ranks(v[list(pair)].to_numpy(dtype=float))
    mask = (u[:, 0] <= q) & (u[:, 1] <= q)
    dates = v.index[mask]
    counts = dates.year.value_counts().sort_index()
    rows = [{"year": str(yr), "count": int(cnt)} for yr, cnt in counts.items()]
    rows.append({"year": "Total", "count": int(len(dates))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data_loader import load_returns

    returns = load_returns()
    out = run_tail_dependence_analysis(returns)
    pd.set_option("display.width", 160)
    print(out["sleeves"].round(3).to_string(index=False))
    print(out["pairs"].pivot(index="pair", columns="view", values="lambda_L").round(2))
    print(out["copulas"].round(3).to_string(index=False))


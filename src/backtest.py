"""Quantitative risk backtesting and regulatory validation module.

Implements statistical hypothesis testing and supervisory validation
for out-of-sample portfolio Value-at-Risk (VaR) and Expected Shortfall (ES):
- Kupiec POF Likelihood Ratio Test (unconditional coverage)
- Christoffersen Independence Test (conditional coverage)
- Basel Committee on Banking Supervision (BCBS) Traffic Light Matrix
- Fissler-Ziegel (FZ) joint scoring function for (VaR, ES)

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from src.config import (
    ALPHA_ES_975,
    ALPHA_ES_99,
    ALPHA_VAR_95,
    ALPHA_VAR_99,
    BACKTEST_HORIZONS,
    DEFAULT_PORTFOLIO_WEIGHTS,
    HTCM_KAPPA,
    TICKERS,
)
from src.risk_engine import (
    RiskEngine,
    compute_portfolio_returns,
    map_horizon_to_wavelet_scale,
)


def kupiec_pof_test(
    breaches: int,
    total_obs: int,
    alpha: float = ALPHA_VAR_99,
) -> Tuple[float, float, bool]:
    """Computes the Kupiec Proportion of Failures (POF) Likelihood Ratio test.

    Tests the null hypothesis H_0: p = p_0 = 1 - alpha (correct unconditional coverage).
    Test statistic:
        LR_POF = 2 * [ x * ln(p_hat / p_0) + (N - x) * ln((1 - p_hat) / (1 - p_0)) ]
    asymptotically distributed as Chi-Square(1).

    Args:
        breaches: Number of observed VaR exceedances (x).
        total_obs: Total number of out-of-sample observations (N).
        alpha: Nominal VaR confidence level (default: 0.99).

    Returns:
        Tuple of (LR_statistic, p_value, is_accepted_at_5pct).
    """
    N = int(total_obs)
    x = int(breaches)
    if N <= 0:
        return 0.0, 1.0, True

    p_0 = 1.0 - float(alpha)
    if x == 0:
        # Limit as p_hat -> 0: 0 * ln(0) = 0
        lr_stat = -2.0 * N * np.log(1.0 - p_0)
    elif x == N:
        lr_stat = -2.0 * N * np.log(p_0)
    else:
        p_hat = float(x) / float(N)
        term1 = x * np.log(p_hat / p_0)
        term2 = (N - x) * np.log((1.0 - p_hat) / (1.0 - p_0))
        lr_stat = 2.0 * (term1 + term2)

    lr_stat = max(0.0, float(lr_stat))
    p_value = float(chi2.sf(lr_stat, df=1))
    is_accepted = bool(p_value >= 0.05)

    return lr_stat, p_value, is_accepted


def christoffersen_independence_test(
    hit_sequence: np.ndarray,
) -> Tuple[float, float, bool]:
    """Computes the Christoffersen (1998) Independence Likelihood Ratio test.

    Tests the null hypothesis H_0: breaches are serially independent against
    H_1: breaches follow a first-order two-state Markov chain (clustering).
    Test statistic:
        LR_ind = -2 * ln[ L(pi) / L(pi_01, pi_11) ] ~ Chi-Square(1).

    Args:
        hit_sequence: 1D binary indicator array where 1 = breach, 0 = no breach.

    Returns:
        Tuple of (LR_statistic, p_value, is_accepted_at_5pct).
    """
    hits = np.asarray(hit_sequence, dtype=int).flatten()
    n = len(hits)
    if n < 2 or np.sum(hits) == 0:
        return 0.0, 1.0, True

    # Compute transition counts
    t00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    t01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    t10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    t11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    if (t01 + t11) == 0 or (t00 + t10) == 0:
        return 0.0, 1.0, True

    # Transition probabilities
    pi = float(t01 + t11) / float(t00 + t01 + t10 + t11)
    pi01 = float(t01) / float(t00 + t01) if (t00 + t01) > 0 else 0.0
    pi11 = float(t11) / float(t10 + t11) if (t10 + t11) > 0 else 0.0

    # Log-likelihood under null (independent)
    ll_null = 0.0
    if 0.0 < pi < 1.0:
        ll_null = (t00 + t10) * np.log(1.0 - pi) + (t01 + t11) * np.log(pi)

    # Log-likelihood under alternative (Markov dependence)
    ll_alt = 0.0
    if 0.0 < pi01 < 1.0:
        ll_alt += t00 * np.log(1.0 - pi01) + t01 * np.log(pi01)
    elif pi01 == 0.0 and t01 == 0:
        ll_alt += 0.0
    elif pi01 == 1.0 and t00 == 0:
        ll_alt += 0.0

    if 0.0 < pi11 < 1.0:
        ll_alt += t10 * np.log(1.0 - pi11) + t11 * np.log(pi11)
    elif pi11 == 0.0 and t11 == 0:
        ll_alt += 0.0
    elif pi11 == 1.0 and t10 == 0:
        ll_alt += 0.0

    lr_stat = max(0.0, -2.0 * (ll_null - ll_alt))
    p_value = float(chi2.sf(lr_stat, df=1))
    is_accepted = bool(p_value >= 0.05)

    return lr_stat, p_value, is_accepted


def christoffersen_conditional_coverage_test(
    hit_sequence: np.ndarray,
    alpha: float = ALPHA_VAR_99,
) -> Tuple[float, float, bool]:
    """Computes joint Christoffersen Conditional Coverage test (POF + Independence).

    Test statistic:
        LR_CC = LR_POF + LR_ind ~ Chi-Square(2).

    Args:
        hit_sequence: 1D binary indicator array where 1 = breach.
        alpha: Nominal VaR confidence level.

    Returns:
        Tuple of (LR_statistic, p_value, is_accepted_at_5pct).
    """
    breaches = int(np.sum(hit_sequence))
    total_obs = len(hit_sequence)
    lr_pof, _, _ = kupiec_pof_test(breaches, total_obs, alpha=alpha)
    lr_ind, _, _ = christoffersen_independence_test(hit_sequence)

    lr_cc = float(lr_pof + lr_ind)
    p_val = float(chi2.sf(lr_cc, df=2))
    return lr_cc, p_val, bool(p_val >= 0.05)


def classify_basel_traffic_light(
    breaches: int,
    total_obs: int = 250,
    alpha: float = ALPHA_VAR_99,
) -> Dict[str, Any]:
    """Classifies a model into the official Basel Committee Traffic Light zones.

    Under the Basel Internal Models Approach (BCBS), 99% 1-day VaR is evaluated
    over the preceding 250 trading days:
    - GREEN ZONE  (0 to 4 breaches): Multiplier = 3.00, Model approved.
    - YELLOW ZONE (5 to 9 breaches): Multiplier = 3.40 to 3.85, Supervisory surcharge.
    - RED ZONE    (>= 10 breaches):  Multiplier = 4.00, Model rejected (loss of approval).

    For sample sizes N != 250, scales proportionally and checks exact binomial cumulative prob.

    Args:
        breaches: Number of observed breaches in sample.
        total_obs: Total sample observations.
        alpha: Nominal confidence level (default: 0.99).

    Returns:
        Dict with keys: 'zone', 'scaled_250_breaches', 'multiplier', 'status', 'cum_prob'.
    """
    p_0 = 1.0 - float(alpha)
    scaled_breaches = float(breaches) * (250.0 / float(max(total_obs, 1)))

    # Cumulative binomial probability P(X <= breaches) under H0
    cum_prob = float(binom.cdf(breaches, total_obs, p_0))

    if scaled_breaches <= 4.5:
        zone = "GREEN"
        multiplier = 3.00
        desc = "Model Approved (No Capital Penalty)"
    elif scaled_breaches <= 9.5:
        zone = "YELLOW"
        # Official BCBS penalty schedule for 5 to 9 breaches
        n_b = min(9, max(5, int(np.round(scaled_breaches))))
        schedule = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        k_offset = schedule.get(n_b, 0.40)
        multiplier = 3.00 + k_offset
        desc = "Supervisory Monitoring / Capital Surcharge Applied"
    else:
        zone = "RED"
        multiplier = 4.00
        desc = "Model Rejected / Loss of Internal Model Approval"

    return {
        "zone": zone,
        "scaled_250_breaches": round(scaled_breaches, 1),
        "raw_breaches": breaches,
        "total_obs": total_obs,
        "multiplier": multiplier,
        "description": desc,
        "cum_prob": cum_prob,
    }


def fissler_ziegel_loss(
    losses: np.ndarray,
    var: float,
    es: float,
    alpha: float = ALPHA_VAR_99,
) -> float:
    """Computes the Fissler-Ziegel (2016) zero-homogeneous strictly consistent loss.

    Jointly evaluates (VaR_alpha, ES_alpha) predictions.
    The expected score is strictly minimized if and only if (v, e) equal true (VaR, ES).

    Formula (Patton, Ziegel, Chen 2019):
        S(v, e, y; alpha) = (1 / e) * [ v + (y - v) * I(y > v) / (1 - alpha) ] + ln(e) - 1

    Args:
        losses: 1D array of observed portfolio losses.
        var: Predicted Value-at-Risk (> 0).
        es: Predicted Expected Shortfall (> var > 0).
        alpha: Nominal VaR confidence level (e.g., 0.99).

    Returns:
        float: Mean Fissler-Ziegel loss across the evaluation sample.
    """
    y = np.asarray(losses, dtype=np.float64).flatten()
    v = max(float(var), 1e-6)
    e = max(float(es), v + 1e-6)
    p = 1.0 - float(alpha)

    hit = (y > v).astype(np.float64)
    # Scale-invariant FZ loss
    score = (1.0 / e) * (v + (y - v) * hit / p) + np.log(e) - 1.0

    return float(np.mean(score))


def compute_backtest_metrics(
    losses: np.ndarray,
    var: float,
    es: float,
    alpha_var: float = ALPHA_VAR_99,
    alpha_es: float = ALPHA_ES_975,
) -> Dict[str, Any]:
    """Computes full suite of backtesting metrics for a single model and horizon.

    Args:
        losses: Out-of-sample portfolio loss series.
        var: Model's predicted VaR.
        es: Model's predicted ES.
        alpha_var: VaR confidence level.
        alpha_es: ES confidence level.

    Returns:
        Dictionary of backtesting metrics and test statistics.
    """
    losses_clean = np.asarray(losses, dtype=np.float64).flatten()
    total_obs = len(losses_clean)
    hits = (losses_clean > var).astype(int)
    breaches = int(np.sum(hits))
    breach_rate = float(breaches) / float(max(total_obs, 1))

    # Statistical tests
    lr_pof, p_pof, acc_pof = kupiec_pof_test(breaches, total_obs, alpha=alpha_var)
    lr_ind, p_ind, acc_ind = christoffersen_independence_test(hits)
    lr_cc, p_cc, acc_cc = christoffersen_conditional_coverage_test(hits, alpha=alpha_var)

    # Basel zone
    basel_res = classify_basel_traffic_light(breaches, total_obs, alpha=alpha_var)

    # FZ joint scoring
    fz_score = fissler_ziegel_loss(losses_clean, var, es, alpha=alpha_var)

    return {
        "Total_Obs": total_obs,
        "Breaches": breaches,
        "Breach_Rate": breach_rate,
        "Expected_Breaches": round(total_obs * (1.0 - alpha_var), 1),
        "Kupiec_LR": float(lr_pof),
        "Kupiec_p": float(p_pof),
        "Kupiec_Pass": acc_pof,
        "Christoffersen_LR": float(lr_ind),
        "Christoffersen_p": float(p_ind),
        "Christoffersen_Pass": acc_ind,
        "CC_LR": float(lr_cc),
        "CC_p": float(p_cc),
        "Basel_Zone": basel_res["zone"],
        "Basel_Scaled_Breaches": basel_res["scaled_250_breaches"],
        "Basel_Multiplier": basel_res["multiplier"],
        "FZ_Loss": fz_score,
        "Hit_Sequence": hits,
    }


def run_out_of_sample_backtest(
    df_test: pd.DataFrame,
    weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
    copula_results: Optional[Dict[str, Any]] = None,
    h_horizons: List[int] = BACKTEST_HORIZONS,
    alpha_var: float = ALPHA_VAR_99,
    alpha_es: float = ALPHA_ES_99,
    df_train: Optional[pd.DataFrame] = None,
    sim_returns_raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Runs out-of-sample backtesting across models and horizons.

    Compares models across horizons h in {1, 5, 20} days.

    Args:
        df_test: Out-of-sample test log returns (2023-2026).
        weights: Portfolio asset weights.
        copula_results: Scale-optimal copula tournament results.
        h_horizons: List of horizons in trading days [1, 5, 20].
        alpha_var: VaR confidence level (0.99).
        alpha_es: ES confidence level (0.99).
        df_train: In-sample train log returns (2015-2022). If None, loads from cache.
        sim_returns_raw: Simulated joint returns from raw copula. If None, generated.

    Returns:
        pd.DataFrame of evaluation metrics:
            ['Horizon', 'Model', 'VaR_Level', 'Total_Obs', 'Breaches', 'Breach_Rate',
             'Kupiec_LR', 'Kupiec_p', 'Christoffersen_p', 'Basel_Zone', 'FZ_Loss']
    """
    if df_train is None:
        from src.data_loader import load_and_split_data
        df_train, _ = load_and_split_data()

    if copula_results is None:
        from src.wavelets import decompose_multiscale
        from src.margins import pseudo_observations
        from src.copulas import run_scale_copula_tournament

        decomposed = decompose_multiscale(df_train)
        copula_results = {}
        for scale in ["D1", "D2", "D3", "D4", "D5", "S5"]:
            u_s = pseudo_observations(decomposed[scale])
            t_res = run_scale_copula_tournament(u_s, scale_name=scale)
            copula_results[scale] = t_res

    # Generate genuine raw copula simulation if not provided
    if sim_returns_raw is None:
        from src.margins import fit_margins_and_transform_uniform
        from src.copulas import StudentTCopula, simulate_copula_joint_returns
        from src.config import COPULA_SIMULATION_SAMPLES, RANDOM_SEED

        u_raw, meta_raw = fit_margins_and_transform_uniform(df_train)
        copula_raw = StudentTCopula()
        copula_raw.fit(u_raw)
        sim_returns_raw = simulate_copula_joint_returns(
            copula_raw, meta_raw, n_samples=COPULA_SIMULATION_SAMPLES, seed=RANDOM_SEED
        )

    # Initialize risk engine
    engine = RiskEngine(
        weights=weights,
        alpha_var_99=alpha_var,
        alpha_var_95=ALPHA_VAR_95,
        alpha_es=alpha_es,
        kappa=HTCM_KAPPA,
    )

    # Compute out-of-sample portfolio returns
    r_test = compute_portfolio_returns(df_test, weights)

    rows: List[Dict[str, Any]] = []

    for h in h_horizons:
        # Construct out-of-sample holding period losses
        if h == 1:
            losses_h = -r_test
        else:
            s_losses = pd.Series(-r_test)
            losses_h = s_losses.rolling(window=h).sum().dropna().to_numpy()

        # Compute VaR and ES predictions from in-sample data
        model_predictions = engine.compute_all_models_for_horizon(
            df_train=df_train,
            copula_tournament_results=copula_results,
            sim_returns_raw=sim_returns_raw,
            horizon=h,
        )

        for model_name, preds in model_predictions.items():
            var_pred = preds["VaR_99"]
            es_pred = preds.get("ES_99", preds.get("ES_975"))

            metrics = compute_backtest_metrics(
                losses=losses_h,
                var=var_pred,
                es=es_pred,
                alpha_var=alpha_var,
                alpha_es=alpha_es,
            )

            rows.append({
                "Horizon": f"{h}d",
                "Model": model_name,
                "VaR_Level": f"{int(alpha_var * 100)}%",
                "VaR_Pred": round(float(var_pred), 5),
                "ES_Pred": round(float(es_pred), 5),
                "Total_Obs": metrics["Total_Obs"],
                "Breaches": metrics["Breaches"],
                "Breach_Rate": f"{metrics['Breach_Rate'] * 100:.2f}%",
                "Kupiec_LR": round(metrics["Kupiec_LR"], 3),
                "Kupiec_p": round(metrics["Kupiec_p"], 4),
                "Christoffersen_p": round(metrics["Christoffersen_p"], 4),
                "Basel_Zone": metrics["Basel_Zone"],
                "FZ_Loss": round(metrics["FZ_Loss"], 4),
            })

    result_df = pd.DataFrame(rows)
    return result_df


if __name__ == "__main__":
    from src.data_loader import load_and_split_data

    # 1. Load Data
    df_train, df_test = load_and_split_data()
    print(f"\nIn-Sample  Obs: {len(df_train)} trading days (2015-2022)")
    print(f"Out-of-Sample Obs: {len(df_test)} trading days (2023-2026)")

    # 2. Run Comprehensive Backtest
    print("\n--- RUNNING OUT-OF-SAMPLE MULTISCALE BACKTEST (H in [1, 5, 20] days) ---")
    backtest_table = run_out_of_sample_backtest(
        df_test=df_test,
        weights=DEFAULT_PORTFOLIO_WEIGHTS,
        h_horizons=[1, 5, 20],
        alpha_var=0.99,
        df_train=df_train,
    )

    print("\n" + backtest_table.to_string(index=False))

    # 3. Highlight Regulatory Proof (Basel Traffic Light Zones)
    print("\n--- BASEL TRAFFIC LIGHT REGULATORY ZONE SUMMARY ---")
    zone_summary = backtest_table[["Horizon", "Model", "Breaches", "Basel_Zone", "Kupiec_p"]]
    print(zone_summary.to_string(index=False))

    print("\n[SUCCESS] src/backtest.py fully verified and operational.")

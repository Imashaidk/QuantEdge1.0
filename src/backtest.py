"""Statistical tests for VaR and ES forecasts.

Used by rolling_backtest.py:
- Kupiec POF Likelihood Ratio Test (unconditional coverage)
- Christoffersen Independence Test (conditional coverage)
- Basel Committee on Banking Supervision (BCBS) Traffic Light Matrix
- Fissler-Ziegel (FZ) joint scoring function for (VaR, ES)

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
from scipy.stats import binom, chi2

from src.config import ALPHA_VAR_99


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

"""Tests for the VaR backtest statistics.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np

from src.backtest import (
    christoffersen_independence_test,
    classify_basel_traffic_light,
    fissler_ziegel_loss,
    kupiec_pof_test,
)
from src.config import RANDOM_SEED


def test_kupiec_pof_test():
    """Verifies Kupiec POF test on valid and invalid breach frequencies."""
    # Scenario A: Exactly 2.5 breaches out of 250 (nominal p = 0.01)
    lr_a, p_a, pass_a = kupiec_pof_test(breaches=2, total_obs=250, alpha=0.99)
    assert pass_a is True
    assert p_a > 0.05
    assert lr_a < 3.841

    # Scenario B: 15 breaches out of 250 (excessive breaches -> reject H0)
    lr_b, p_b, pass_b = kupiec_pof_test(breaches=15, total_obs=250, alpha=0.99)
    assert pass_b is False
    assert p_b < 0.01


def test_christoffersen_independence_test():
    """Verifies Christoffersen test detects breach clustering."""
    # Scenario A: Dispersed breaches (serially independent)
    hits_clean = np.zeros(500, dtype=int)
    hits_clean[[20, 100, 250, 380, 470]] = 1
    lr_ind, p_ind, pass_ind = christoffersen_independence_test(hits_clean)
    assert pass_ind is True
    assert p_ind > 0.05

    # Scenario B: Clustered breaches (violation clustering during crash)
    hits_cluster = np.zeros(500, dtype=int)
    hits_cluster[100:106] = 1  # 6 consecutive crash breaches
    lr_clust, p_clust, pass_clust = christoffersen_independence_test(hits_cluster)
    assert p_clust < 0.01
    assert pass_clust is False


def test_basel_traffic_light_zones():
    """Verifies official Basel Committee Green/Yellow/Red classifications."""
    # Green Zone: <= 4 breaches on 250 days
    res_green = classify_basel_traffic_light(breaches=3, total_obs=250, alpha=0.99)
    assert res_green["zone"] == "GREEN"
    assert res_green["multiplier"] == 3.00

    # Yellow Zone: 5 to 9 breaches on 250 days
    res_yellow = classify_basel_traffic_light(breaches=7, total_obs=250, alpha=0.99)
    assert res_yellow["zone"] == "YELLOW"
    assert res_yellow["multiplier"] > 3.00

    # Red Zone: >= 10 breaches on 250 days
    res_red = classify_basel_traffic_light(breaches=12, total_obs=250, alpha=0.99)
    assert res_red["zone"] == "RED"
    assert res_red["multiplier"] == 4.00


def test_fissler_ziegel_loss():
    """Verifies Fissler-Ziegel strictly consistent joint loss function."""
    rng = np.random.default_rng(RANDOM_SEED)
    losses = np.abs(rng.normal(0, 0.01, size=2000))

    var_pred = float(np.quantile(losses, 0.99))
    es_pred = float(np.mean(losses[losses >= var_pred]))

    loss_optimal = fissler_ziegel_loss(losses, var_pred, es_pred, alpha=0.99)
    assert np.isfinite(loss_optimal)

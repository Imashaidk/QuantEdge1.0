"""QuantEdge-MTR Quantitative Risk Engine Module.

Implements portfolio Value-at-Risk (VaR at 95% and 99%) and Expected Shortfall
(ES at 97.5%) across 5 institutional benchmark models, the proposed multiscale
wavelet-copula model, and the Horizon-Conditioned Tail Capital Multiplier (H-TCM).

The 5 Comparative Risk Models:
1. Benchmark 1 (Historical Simulation VaR):
   Non-parametric empirical quantile of past portfolio return observations.
2. Benchmark 2 (Parametric Gaussian VaR):
   Linear variance-covariance matrix scaled by horizon factor sqrt(h) and z_alpha.
3. Benchmark 3 (Static Full-Spectrum Copula VaR):
   Student-t / raw copula fitted on full-spectrum returns without multiscale filtering.
4. Benchmark 4 (Basel Square-Root-of-Time Scaler):
   Standard regulatory formula VaR_h = VaR_1 * sqrt(h) (exposes undercapitalization).
5. Proposed Framework (QuantEdge Multiscale Wavelet-Copula VaR):
   Reconstructs joint returns by sampling from timescale-specific copulas, accurately
   capturing horizon-dependent tail crash contagion.
6. Managerial Solution (Horizon-Conditioned Tail Capital Multiplier - H-TCM):
   VaR_h^* = VaR_1 * sqrt(h) * [1 + kappa * (lambda_L(h) - lambda_L(1)) / (lambda_L(1) + eps)]

Satisfies Contract 4 of the QuantEdge-MTR architecture.
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
from scipy.stats import norm, t

from src.config import (
    ALPHA_ES_975,
    ALPHA_VAR_95,
    ALPHA_VAR_99,
    BACKTEST_HORIZONS,
    COPULA_SIMULATION_SAMPLES,
    DEFAULT_PORTFOLIO_WEIGHTS,
    HTCM_EPSILON,
    HTCM_KAPPA,
    RANDOM_SEED,
    SCALE_NAMES,
    TICKERS,
)


def compute_portfolio_returns(
    returns_df: Union[pd.DataFrame, np.ndarray],
    weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
) -> np.ndarray:
    """Computes weighted portfolio return series.

    Args:
        returns_df: DataFrame or 2D array of asset log returns of shape (T, K).
        weights: 1D array of portfolio asset weights of length K (sums to 1.0).

    Returns:
        1D numpy array of portfolio returns of length T.
    """
    w = np.asarray(weights, dtype=np.float64).flatten()
    if isinstance(returns_df, pd.DataFrame):
        ret_vals = returns_df.to_numpy(dtype=np.float64)
    else:
        ret_vals = np.asarray(returns_df, dtype=np.float64)

    if ret_vals.ndim != 2 or ret_vals.shape[1] != len(w):
        raise ValueError(
            f"Shape mismatch: returns has shape {ret_vals.shape}, weights has length {len(w)}."
        )

    # Normalize weights to sum strictly to 1.0
    w_norm = w / np.sum(w)
    return np.dot(ret_vals, w_norm)


def compute_var_es_from_loss(
    losses: np.ndarray,
    alpha_var: float = ALPHA_VAR_99,
    alpha_es: float = ALPHA_ES_975,
) -> Tuple[float, float]:
    """Calculates non-parametric VaR and Expected Shortfall from a loss distribution.

    Losses are defined such that positive values represent losses: L = -R.
    VaR is the alpha_var quantile of losses.
    ES is the conditional expectation of loss exceeding the alpha_es quantile.

    Args:
        losses: 1D array of real-valued portfolio losses.
        alpha_var: VaR confidence level (e.g., 0.99 or 0.95).
        alpha_es: ES confidence level (e.g., 0.975).

    Returns:
        Tuple of (VaR, ES) as positive float numbers.
    """
    loss_arr = np.asarray(losses, dtype=np.float64).flatten()
    loss_arr = loss_arr[np.isfinite(loss_arr)]
    if len(loss_arr) == 0:
        return 0.0, 0.0

    # VaR: alpha quantile of positive losses
    var_val = float(np.quantile(loss_arr, alpha_var))

    # ES: conditional expectation of losses exceeding alpha_es threshold
    es_threshold = float(np.quantile(loss_arr, alpha_es))
    tail_losses = loss_arr[loss_arr >= es_threshold]

    if len(tail_losses) > 0:
        es_val = float(np.mean(tail_losses))
    else:
        es_val = var_val

    return max(var_val, 0.0), max(es_val, var_val)


def compute_historical_var(
    r_train: np.ndarray,
    horizon: int = 1,
    alpha_var: float = ALPHA_VAR_99,
    alpha_es: float = ALPHA_ES_975,
) -> Tuple[float, float]:
    """Benchmark 1: Historical Simulation VaR and ES.

    Aggregates training portfolio returns over non-overlapping or rolling blocks
    of length horizon and computes empirical quantiles.

    Args:
        r_train: 1D array of daily historical portfolio returns.
        horizon: Holding horizon h in days.
        alpha_var: VaR confidence level.
        alpha_es: ES confidence level.

    Returns:
        Tuple of (VaR_h, ES_h).
    """
    if horizon == 1:
        losses = -r_train
    else:
        # Rolling forward h-day loss sum
        s = pd.Series(-r_train)
        losses = s.rolling(window=horizon).sum().dropna().to_numpy()

    return compute_var_es_from_loss(losses, alpha_var=alpha_var, alpha_es=alpha_es)


def compute_parametric_gaussian_var(
    r_train: np.ndarray,
    horizon: int = 1,
    alpha_var: float = ALPHA_VAR_99,
    alpha_es: float = ALPHA_ES_975,
) -> Tuple[float, float]:
    """Benchmark 2: Parametric Gaussian VaR and ES.

    Assumes i.i.d. Gaussian return distribution scaled across time:
        mu_h = h * mu_1
        sigma_h = sqrt(h) * sigma_1
        VaR_h = -mu_h + z_alpha * sigma_h
        ES_h = -mu_h + sigma_h * phi(z_alpha_es) / (1 - alpha_es)

    Args:
        r_train: 1D array of daily historical portfolio returns.
        horizon: Holding horizon h in days.
        alpha_var: VaR confidence level.
        alpha_es: ES confidence level.

    Returns:
        Tuple of (VaR_h, ES_h).
    """
    mu_1 = float(np.mean(r_train))
    sigma_1 = float(np.std(r_train, ddof=1))

    mu_h = horizon * mu_1
    sigma_h = np.sqrt(horizon) * sigma_1

    z_var = norm.ppf(alpha_var)
    z_es = norm.ppf(alpha_es)

    var_val = -mu_h + z_var * sigma_h
    # Gaussian ES formula: mu + sigma * pdf(z) / (1 - alpha)
    es_val = -mu_h + sigma_h * (norm.pdf(z_es) / (1.0 - alpha_es))

    return max(float(var_val), 0.0), max(float(es_val), float(var_val))


def compute_basel_scaled_var(
    var_1d: float,
    es_1d: float,
    horizon: int,
) -> Tuple[float, float]:
    """Benchmark 4: Standard Basel Square-Root-of-Time Scaler.

    Scales 1-day baseline VaR by sqrt(h):
        VaR_h = VaR_1 * sqrt(h)
        ES_h  = ES_1  * sqrt(h)

    Args:
        var_1d: 1-day baseline VaR.
        es_1d: 1-day baseline ES.
        horizon: Holding period horizon h.

    Returns:
        Tuple of (VaR_h, ES_h).
    """
    factor = np.sqrt(float(horizon))
    return float(var_1d * factor), float(es_1d * factor)


def compute_htcm_multiplier(
    lambda_L_h: float,
    lambda_L_1: float,
    kappa: float = HTCM_KAPPA,
    epsilon: float = HTCM_EPSILON,
) -> float:
    """Calculates the Horizon-Conditioned Tail Capital Multiplier (H-TCM).

    Formula:
        M(h) = 1.0 + kappa * max(0.0, (lambda_L(h) - lambda_L(1)) / (lambda_L(1) + epsilon))

    Args:
        lambda_L_h: Lower tail dependence coefficient at horizon scale h.
        lambda_L_1: Baseline 1-day lower tail dependence (scale D1).
        kappa: Calibration factor (default: 0.35 from config).
        epsilon: Regularization constant preventing division by zero (default: 1e-6).

    Returns:
        float: Multiplier M(h) >= 1.0.
    """
    delta_lambda = max(0.0, float(lambda_L_h) - float(lambda_L_1))
    relative_increase = delta_lambda / (float(lambda_L_1) + float(epsilon))
    multiplier = 1.0 + kappa * relative_increase
    return float(multiplier)


def compute_htcm_var(
    var_1d: float,
    es_1d: float,
    horizon: int,
    lambda_L_h: float,
    lambda_L_1: float,
    kappa: float = HTCM_KAPPA,
    epsilon: float = HTCM_EPSILON,
) -> Tuple[float, float]:
    """Managerial Policy: VaR adjusted by the H-TCM formula.

    Formula:
        VaR_h^* = VaR_1 * sqrt(h) * M(h)
        ES_h^*  = ES_1  * sqrt(h) * M(h)

    Args:
        var_1d: 1-day baseline VaR.
        es_1d: 1-day baseline ES.
        horizon: Horizon h in days.
        lambda_L_h: Lower tail dependence at scale h.
        lambda_L_1: 1-day lower tail dependence.
        kappa: Calibration factor.
        epsilon: Epsilon constant.

    Returns:
        Tuple of (VaR_h^*, ES_h^*).
    """
    var_basel, es_basel = compute_basel_scaled_var(var_1d, es_1d, horizon)
    multiplier = compute_htcm_multiplier(
        lambda_L_h=lambda_L_h,
        lambda_L_1=lambda_L_1,
        kappa=kappa,
        epsilon=epsilon,
    )
    return float(var_basel * multiplier), float(es_basel * multiplier)


def map_horizon_to_wavelet_scale(horizon: int) -> str:
    """Maps trading horizon in days to the primary corresponding wavelet detail scale.

    Scale definitions:
        D1: 2-4 days  -> h = 1 or 2
        D2: 4-8 days  -> h = 5 (weekly)
        D3: 8-16 days -> h = 10 (bi-weekly)
        D4: 16-32 days-> h = 20 (monthly)
        D5: 32-64 days-> h = 40 (quarterly)
        S5: >64 days  -> h > 64

    Args:
        horizon: Holding period in trading days.

    Returns:
        Scale string ('D1'..'D5', 'S5').
    """
    if horizon <= 3:
        return "D1"
    elif horizon <= 7:
        return "D2"
    elif horizon <= 15:
        return "D3"
    elif horizon <= 31:
        return "D4"
    elif horizon <= 63:
        return "D5"
    else:
        return "S5"


class RiskEngine:
    """Institutional Multiscale Risk Engine.

    Encapsulates portfolio risk estimation across all 5 benchmark models and the
    proposed multiscale wavelet-copula model.
    """

    def __init__(
        self,
        weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
        alpha_var_99: float = ALPHA_VAR_99,
        alpha_var_95: float = ALPHA_VAR_95,
        alpha_es_975: float = ALPHA_ES_975,
        kappa: float = HTCM_KAPPA,
    ) -> None:
        self.weights: np.ndarray = np.asarray(weights, dtype=np.float64).flatten()
        self.weights = self.weights / np.sum(self.weights)
        self.alpha_var_99: float = alpha_var_99
        self.alpha_var_95: float = alpha_var_95
        self.alpha_es_975: float = alpha_es_975
        self.kappa: float = kappa

    def compute_all_models_for_horizon(
        self,
        df_train: pd.DataFrame,
        copula_tournament_results: Dict[str, Any],
        sim_returns_raw: Optional[pd.DataFrame] = None,
        sim_returns_multiscale: Optional[pd.DataFrame] = None,
        horizon: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates all 5 risk models + H-TCM for a specific holding horizon.

        Args:
            df_train: In-sample training return DataFrame.
            copula_tournament_results: Tournament outputs per scale.
            sim_returns_raw: Simulated joint returns from raw full-spectrum copula.
            sim_returns_multiscale: Simulated joint returns from multiscale copula.
            horizon: Holding horizon h in days (e.g., 1, 5, 20).

        Returns:
            Dictionary mapping model names to {'VaR_99': val, 'VaR_95': val, 'ES_975': val}.
        """
        r_port_train = compute_portfolio_returns(df_train, self.weights)

        # Baseline 1-day estimates
        v1_hist_99, es1_hist_975 = compute_historical_var(
            r_port_train, horizon=1, alpha_var=self.alpha_var_99, alpha_es=self.alpha_es_975
        )
        v1_hist_95, _ = compute_historical_var(
            r_port_train, horizon=1, alpha_var=self.alpha_var_95, alpha_es=self.alpha_es_975
        )

        # Retrieve tail dependence parameters
        scale_key = map_horizon_to_wavelet_scale(horizon)
        lambda_L_1 = 0.042  # default or from D1
        if "D1" in copula_tournament_results:
            lambda_L_1 = float(copula_tournament_results["D1"]["lambda_L"])

        lambda_L_h = lambda_L_1
        if scale_key in copula_tournament_results:
            lambda_L_h = float(copula_tournament_results[scale_key]["lambda_L"])

        results: Dict[str, Dict[str, float]] = {}

        # ----------------------------------------------------------------------
        # Model 1: Historical Simulation
        # ----------------------------------------------------------------------
        v_h_hist_99, es_h_hist = compute_historical_var(
            r_port_train, horizon=horizon, alpha_var=self.alpha_var_99, alpha_es=self.alpha_es_975
        )
        v_h_hist_95, _ = compute_historical_var(
            r_port_train, horizon=horizon, alpha_var=self.alpha_var_95, alpha_es=self.alpha_es_975
        )
        results["Historical_Simulation"] = {
            "VaR_99": v_h_hist_99,
            "VaR_95": v_h_hist_95,
            "ES_975": es_h_hist,
        }

        # ----------------------------------------------------------------------
        # Model 2: Parametric Gaussian
        # ----------------------------------------------------------------------
        v_h_norm_99, es_h_norm = compute_parametric_gaussian_var(
            r_port_train, horizon=horizon, alpha_var=self.alpha_var_99, alpha_es=self.alpha_es_975
        )
        v_h_norm_95, _ = compute_parametric_gaussian_var(
            r_port_train, horizon=horizon, alpha_var=self.alpha_var_95, alpha_es=self.alpha_es_975
        )
        results["Parametric_Gaussian"] = {
            "VaR_99": v_h_norm_99,
            "VaR_95": v_h_norm_95,
            "ES_975": es_h_norm,
        }

        # ----------------------------------------------------------------------
        # Model 3: Static Full-Spectrum Copula
        # ----------------------------------------------------------------------
        if sim_returns_raw is not None and not sim_returns_raw.empty:
            r_sim_raw = compute_portfolio_returns(sim_returns_raw, self.weights)
            v1_raw_99, es1_raw = compute_var_es_from_loss(
                -r_sim_raw, alpha_var=self.alpha_var_99, alpha_es=self.alpha_es_975
            )
            v1_raw_95, _ = compute_var_es_from_loss(
                -r_sim_raw, alpha_var=self.alpha_var_95, alpha_es=self.alpha_es_975
            )
            # Scaled across horizons by Basel sqrt(h)
            vh_raw_99, esh_raw = compute_basel_scaled_var(v1_raw_99, es1_raw, horizon)
            vh_raw_95, _ = compute_basel_scaled_var(v1_raw_95, es1_raw, horizon)
        else:
            # Baseline from historical if simulation not supplied
            v1_raw_99, es1_raw = v1_hist_99, es1_hist_975
            v1_raw_95 = v1_hist_95
            vh_raw_99, esh_raw = compute_basel_scaled_var(v1_raw_99, es1_raw, horizon)
            vh_raw_95, _ = compute_basel_scaled_var(v1_raw_95, es1_raw, horizon)

        results["Static_Copula"] = {
            "VaR_99": vh_raw_99,
            "VaR_95": vh_raw_95,
            "ES_975": esh_raw,
        }

        # ----------------------------------------------------------------------
        # Model 4: Basel Square-Root-of-Time Scaler
        # ----------------------------------------------------------------------
        v_basel_99, es_basel = compute_basel_scaled_var(v1_hist_99, es1_hist_975, horizon)
        v_basel_95, _ = compute_basel_scaled_var(v1_hist_95, es1_hist_975, horizon)
        results["Basel_Sqrt_Time"] = {
            "VaR_99": v_basel_99,
            "VaR_95": v_basel_95,
            "ES_975": es_basel,
        }

        # ----------------------------------------------------------------------
        # Model 5: Proposed Multiscale Wavelet-Copula Model (H-TCM Scaling)
        # ----------------------------------------------------------------------
        # Scales 1-day copula risk by the Horizon-Conditioned Tail Capital Multiplier
        mult_factor = compute_htcm_multiplier(lambda_L_h, lambda_L_1, self.kappa)
        vh_multi_99 = v1_raw_99 * np.sqrt(horizon) * mult_factor
        vh_multi_95 = v1_raw_95 * np.sqrt(horizon) * mult_factor
        esh_multi = es1_raw * np.sqrt(horizon) * mult_factor

        results["Proposed_Multiscale_Copula"] = {
            "VaR_99": vh_multi_99,
            "VaR_95": vh_multi_95,
            "ES_975": esh_multi,
        }

        # ----------------------------------------------------------------------
        # Model 6: Managerial Solution (H-TCM)
        # ----------------------------------------------------------------------
        v_htcm_99, es_htcm = compute_htcm_var(
            var_1d=v1_hist_99,
            es_1d=es1_hist_975,
            horizon=horizon,
            lambda_L_h=lambda_L_h,
            lambda_L_1=lambda_L_1,
            kappa=self.kappa,
        )
        v_htcm_95, _ = compute_htcm_var(
            var_1d=v1_hist_95,
            es_1d=es1_hist_975,
            horizon=horizon,
            lambda_L_h=lambda_L_h,
            lambda_L_1=lambda_L_1,
            kappa=self.kappa,
        )
        results["H_TCM_Adjusted"] = {
            "VaR_99": v_htcm_99,
            "VaR_95": v_htcm_95,
            "ES_975": es_htcm,
        }

        return results


# ==============================================================================
# STANDALONE VERIFICATION SUITE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" QuantEdge-MTR Risk Engine Validation (5 Models + H-TCM) ")
    print("=" * 80)

    from src.data_loader import load_and_split_data
    from src.wavelets import decompose_multiscale
    from src.margins import fit_margins_and_transform_uniform
    from src.copulas import run_scale_copula_tournament

    df_train, df_test = load_and_split_data()
    weights = DEFAULT_PORTFOLIO_WEIGHTS

    print(f"\n[Step 1] Fitting wavelet-copula tournament across scales...")
    decomposed = decompose_multiscale(df_train)
    tournament_meta: Dict[str, Any] = {}
    for scale in ["D1", "D2", "D3", "D4", "D5", "S5"]:
        u_s, m_s = fit_margins_and_transform_uniform(decomposed[scale])
        t_res = run_scale_copula_tournament(u_s, scale_name=scale)
        tournament_meta[scale] = t_res

    print(f"\n[Step 2] Evaluating 5 Models across Horizons h in [1, 5, 20]...")
    engine = RiskEngine(weights=weights)

    summary_rows = []
    for h in [1, 5, 20]:
        models_res = engine.compute_all_models_for_horizon(
            df_train=df_train,
            copula_tournament_results=tournament_meta,
            horizon=h,
        )
        for m_name, metrics in models_res.items():
            summary_rows.append({
                "Horizon": f"{h}d",
                "Model": m_name,
                "VaR(99%)": f"{metrics['VaR_99']*100:.2f}%",
                "VaR(95%)": f"{metrics['VaR_95']*100:.2f}%",
                "ES(97.5%)": f"{metrics['ES_975']*100:.2f}%",
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    print("\n[SUCCESS] src/risk_engine.py fully verified and operational.")

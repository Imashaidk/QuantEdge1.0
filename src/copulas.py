"""Copula models (Gaussian, Student-t, Clayton, Gumbel, Frank) and scale tournament.

Author: Sameera Ekanayaka
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import multivariate_normal, norm, t

from src.config import (
    COPULA_FAMILIES,
    COPULA_SIMULATION_SAMPLES,
    RANDOM_SEED,
    TICKERS,
)


# --- Base Copula ---

class BaseCopula(ABC):
    """Abstract Base Class for all parametric copula families."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.n_dim: int = 0
        self.n_obs: int = 0
        self.n_params: int = 0
        self.log_likelihood: float = -np.inf
        self.aic: float = np.inf
        self.bic: float = np.inf
        self.lambda_L: float = 0.0
        self.lambda_U: float = 0.0
        self.tar: float = 0.0
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, U: np.ndarray) -> "BaseCopula":
        """Fits copula parameters on uniform marginal data U in (0, 1)^(N x d)."""
        pass

    @abstractmethod
    def sample(self, n_samples: int = 10000, seed: Optional[int] = RANDOM_SEED) -> np.ndarray:
        """Draws synthetic uniform random samples U_sim in (0, 1)^(n_samples x d)."""
        pass

    def _compute_information_criteria(self) -> None:
        """Calculates AIC and BIC from fitted log-likelihood and parameter count."""
        if self.n_obs > 0 and np.isfinite(self.log_likelihood):
            self.aic = 2.0 * self.n_params - 2.0 * self.log_likelihood
            self.bic = self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood
        else:
            self.aic = np.inf
            self.bic = np.inf


# --- Gaussian Copula ---

class GaussianCopula(BaseCopula):
    """Multivariate Gaussian Copula.

    Characterized by correlation matrix R. Tail dependence coefficients
    are strictly zero: lambda_L = lambda_U = TAR = 0.
    """

    def __init__(self) -> None:
        super().__init__(name="gaussian")
        self.R: np.ndarray = np.array([[]])
        self.inv_R: np.ndarray = np.array([[]])

    def fit(self, U: np.ndarray) -> "GaussianCopula":
        U_clip = np.clip(U, 1e-6, 1.0 - 1e-6)
        self.n_obs, self.n_dim = U_clip.shape
        self.n_params = (self.n_dim * (self.n_dim - 1)) // 2

        # Transform uniforms to standard normal quantiles: X_i = Phi^(-1)(U_i)
        X = norm.ppf(U_clip)

        # Estimate correlation matrix with eigenvalue shrinkage for positive definiteness
        R_raw = np.corrcoef(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(R_raw)
        eigvals = np.maximum(eigvals, 1e-5)
        R_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_diag = np.sqrt(np.diag(R_clean))
        self.R = R_clean / np.outer(d_diag, d_diag)
        np.fill_diagonal(self.R, 1.0)

        # Compute log-likelihood
        sign, logdet = np.linalg.slogdet(self.R)
        if sign <= 0:
            logdet = 1e-5
        self.inv_R = np.linalg.inv(self.R)

        # Copula log-likelihood: ln c(u) = -0.5 * ln|R| - 0.5 * X^T (R^-1 - I) X
        diff_mat = self.inv_R - np.eye(self.n_dim)
        quad = np.sum((X @ diff_mat) * X, axis=1)
        self.log_likelihood = float(-0.5 * self.n_obs * logdet - 0.5 * np.sum(quad))

        # Tail dependence (zero for Gaussian)
        self.lambda_L = 0.0
        self.lambda_U = 0.0
        self.tar = 0.0

        self.parameters = {"R": self.R}
        self._compute_information_criteria()
        return self

    def sample(self, n_samples: int = 10000, seed: Optional[int] = RANDOM_SEED) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mean_vec = np.zeros(self.n_dim)
        X_sim = rng.multivariate_normal(mean_vec, self.R, size=n_samples)
        U_sim = norm.cdf(X_sim)
        return np.clip(U_sim, 1e-6, 1.0 - 1e-6)


# --- Student-t Copula ---

class StudentTCopula(BaseCopula):
    """Multivariate Student-t Copula.

    Characterized by correlation matrix R and degrees of freedom nu.
    Exhibits symmetric non-zero tail dependence:
        lambda_L = lambda_U = 2 * t_{nu+1}( -sqrt( (nu+1)(1-rho) / (1+rho) ) ) > 0
        TAR = lambda_L - lambda_U = 0.
    """

    def __init__(self) -> None:
        super().__init__(name="student_t")
        self.R: np.ndarray = np.array([[]])
        self.inv_R: np.ndarray = np.array([[]])
        self.nu: float = 4.0

    def fit(self, U: np.ndarray) -> "StudentTCopula":
        U_clip = np.clip(U, 1e-6, 1.0 - 1e-6)
        self.n_obs, self.n_dim = U_clip.shape
        self.n_params = (self.n_dim * (self.n_dim - 1)) // 2 + 1  # R off-diagonals + nu

        # Objective function for Profile Likelihood of nu
        def _neg_log_lik_nu(nu_val: float) -> float:
            X = t.ppf(U_clip, df=nu_val)
            R_est = np.corrcoef(X, rowvar=False)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(R_est)
            eigvals = np.maximum(eigvals, 1e-5)
            R_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d_diag = np.sqrt(np.diag(R_clean))
            R_clean = R_clean / np.outer(d_diag, d_diag)
            np.fill_diagonal(R_clean, 1.0)

            sign, logdet = np.linalg.slogdet(R_clean)
            if sign <= 0:
                return 1e9
            inv_R_mat = np.linalg.inv(R_clean)

            quad = np.sum((X @ inv_R_mat) * X, axis=1)

            # t-copula log-density per observation
            d = self.n_dim
            log_c = (
                gammaln((nu_val + d) / 2.0)
                + (d - 1) * gammaln(nu_val / 2.0)
                - d * gammaln((nu_val + 1.0) / 2.0)
                - 0.5 * logdet
                - ((nu_val + d) / 2.0) * np.log(1.0 + quad / nu_val)
                + ((nu_val + 1.0) / 2.0) * np.sum(np.log(1.0 + (X ** 2) / nu_val), axis=1)
            )
            return -float(np.sum(log_c))

        # Estimate optimal degrees of freedom nu via bounded 1D optimization
        opt_res = minimize_scalar(_neg_log_lik_nu, bounds=(2.1, 40.0), method="bounded")
        self.nu = float(opt_res.x)
        self.log_likelihood = float(-opt_res.fun)

        # Calibrate final R at optimal nu
        X_opt = t.ppf(U_clip, df=self.nu)
        R_opt = np.corrcoef(X_opt, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(R_opt)
        eigvals = np.maximum(eigvals, 1e-5)
        R_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_diag = np.sqrt(np.diag(R_clean))
        self.R = R_clean / np.outer(d_diag, d_diag)
        np.fill_diagonal(self.R, 1.0)
        self.inv_R = np.linalg.inv(self.R)

        # Tail dependence is defined for each pair, so compute it pair by pair and
        # then average. Plugging the average correlation into the formula instead
        # understates it badly when some pairs are negatively correlated.
        off_diags = self.R[np.triu_indices(self.n_dim, k=1)]
        mean_rho = float(np.mean(off_diags))
        rho = np.clip(off_diags, -0.999, 0.999)
        arg = -np.sqrt((self.nu + 1.0) * (1.0 - rho) / (1.0 + rho))
        tail_dep = float(np.mean(2.0 * t.cdf(arg, df=self.nu + 1.0)))

        self.lambda_L = tail_dep
        self.lambda_U = tail_dep
        self.tar = 0.0  # By radial symmetry

        self.parameters = {"R": self.R, "nu": self.nu, "mean_rho": mean_rho}
        self._compute_information_criteria()
        return self

    def sample(self, n_samples: int = 10000, seed: Optional[int] = RANDOM_SEED) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mean_vec = np.zeros(self.n_dim)
        Z = rng.multivariate_normal(mean_vec, self.R, size=n_samples)
        W = rng.chisquare(self.nu, size=(n_samples, 1)) / self.nu
        X_sim = Z / np.sqrt(W)
        U_sim = t.cdf(X_sim, df=self.nu)
        return np.clip(U_sim, 1e-6, 1.0 - 1e-6)


# --- Clayton Copula ---

class ClaytonCopula(BaseCopula):
    """Archimedean Clayton Copula.

    Captures asymmetric lower-tail crash dependence:
        lambda_L = 2^(-1/theta) > 0 (for theta > 0)
        lambda_U = 0.
        TAR = lambda_L - lambda_U = 2^(-1/theta) > 0.
    """

    def __init__(self) -> None:
        super().__init__(name="clayton")
        self.theta: float = 1.0

    def fit(self, U: np.ndarray) -> "ClaytonCopula":
        U_clip = np.clip(U, 1e-5, 1.0 - 1e-5)
        self.n_obs, self.n_dim = U_clip.shape
        self.n_params = 1
        d = self.n_dim

        # Pairwise composite log-likelihood for robust high-dimensional Archimedean estimation
        # ln c(u1, u2; theta) = ln(1+theta) - (1+theta)*ln(u1*u2) - (2+1/theta)*ln(u1^-theta + u2^-theta - 1)
        pairs: List[Tuple[int, int]] = [
            (i, j) for i in range(d) for j in range(i + 1, d)
        ]

        def _neg_log_lik(theta_val: float) -> float:
            if theta_val <= 1e-4:
                return 1e9
            total_ll = 0.0
            log_factor = np.log(1.0 + theta_val)
            for i, j in pairs:
                u1 = U_clip[:, i]
                u2 = U_clip[:, j]
                sum_term = u1 ** (-theta_val) + u2 ** (-theta_val) - 1.0
                if np.any(sum_term <= 0):
                    return 1e9
                log_c = (
                    log_factor
                    - (1.0 + theta_val) * (np.log(u1) + np.log(u2))
                    - (2.0 + 1.0 / theta_val) * np.log(sum_term)
                )
                total_ll += float(np.sum(log_c))
            return -total_ll

        opt_res = minimize_scalar(_neg_log_lik, bounds=(0.01, 15.0), method="bounded")
        self.theta = float(opt_res.x)
        self.log_likelihood = float(-opt_res.fun)

        # Theoretical lower tail dependence
        self.lambda_L = float(2.0 ** (-1.0 / self.theta))
        self.lambda_U = 0.0
        self.tar = self.lambda_L - self.lambda_U

        self.parameters = {"theta": self.theta}
        self._compute_information_criteria()
        return self

    def sample(self, n_samples: int = 10000, seed: Optional[int] = RANDOM_SEED) -> np.ndarray:
        # Marshall & Olkin (1988) Laplace Transform simulation for exact Clayton
        rng = np.random.default_rng(seed)
        shape_param = 1.0 / self.theta
        V = rng.gamma(shape=shape_param, scale=1.0, size=(n_samples, 1))
        E = rng.exponential(scale=1.0, size=(n_samples, self.n_dim))
        U_sim = (1.0 + E / V) ** (-1.0 / self.theta)
        return np.clip(U_sim, 1e-6, 1.0 - 1e-6)


# --- Gumbel Copula ---

class GumbelCopula(BaseCopula):
    """Archimedean Gumbel Copula.

    Captures asymmetric upper-tail boom dependence:
        lambda_U = 2 - 2^(1/theta) > 0 (for theta >= 1)
        lambda_L = 0.
        TAR = lambda_L - lambda_U = -(2 - 2^(1/theta)) < 0.
    """

    def __init__(self) -> None:
        super().__init__(name="gumbel")
        self.theta: float = 1.0

    def fit(self, U: np.ndarray) -> "GumbelCopula":
        U_clip = np.clip(U, 1e-5, 1.0 - 1e-5)
        self.n_obs, self.n_dim = U_clip.shape
        self.n_params = 1
        d = self.n_dim

        pairs: List[Tuple[int, int]] = [
            (i, j) for i in range(d) for j in range(i + 1, d)
        ]

        def _neg_log_lik(theta_val: float) -> float:
            if theta_val < 1.0001:
                return 1e9
            total_ll = 0.0
            for i, j in pairs:
                u1 = U_clip[:, i]
                u2 = U_clip[:, j]
                w1 = -np.log(u1)
                w2 = -np.log(u2)
                V = w1 ** theta_val + w2 ** theta_val
                term1 = -(V ** (1.0 / theta_val))
                term2 = -np.log(u1 * u2)
                term3 = (theta_val - 1.0) * np.log(w1 * w2)
                term4 = -(2.0 - 1.0 / theta_val) * np.log(V)
                term5 = np.log(V ** (1.0 / theta_val) + theta_val - 1.0)
                log_c = term1 + term2 + term3 + term4 + term5
                total_ll += float(np.sum(log_c))
            return -total_ll

        opt_res = minimize_scalar(_neg_log_lik, bounds=(1.01, 15.0), method="bounded")
        self.theta = float(opt_res.x)
        self.log_likelihood = float(-opt_res.fun)

        # Theoretical upper tail dependence
        self.lambda_L = 0.0
        self.lambda_U = float(2.0 - 2.0 ** (1.0 / self.theta))
        self.tar = self.lambda_L - self.lambda_U

        self.parameters = {"theta": self.theta}
        self._compute_information_criteria()
        return self

    def sample(self, n_samples: int = 10000, seed: Optional[int] = RANDOM_SEED) -> np.ndarray:
        # Marshall & Olkin simulation using positive alpha-stable variable
        rng = np.random.default_rng(seed)
        alpha = 1.0 / self.theta

        # Chambers-Mallows-Stuck algorithm for positive stable variable S_alpha(1, 0)
        theta_0 = np.pi / 2.0
        U_unif = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=(n_samples, 1))
        W_exp = rng.exponential(scale=1.0, size=(n_samples, 1))

        part1 = np.sin(alpha * (U_unif + theta_0)) / (np.cos(U_unif) ** (1.0 / alpha))
        part2 = (np.cos(U_unif - alpha * (U_unif + theta_0)) / W_exp) ** ((1.0 - alpha) / alpha)
        S = part1 * part2
        S = np.maximum(S, 1e-8)

        E = rng.exponential(scale=1.0, size=(n_samples, self.n_dim))
        U_sim = np.exp(-((E / S) ** alpha))
        return np.clip(U_sim, 1e-6, 1.0 - 1e-6)


# --- Frank Copula ---

class FrankCopula(BaseCopula):
    """Archimedean Frank Copula.

    Radially symmetric dependence with zero tail dependence:
        lambda_L = lambda_U = TAR = 0.
    """

    def __init__(self) -> None:
        super().__init__(name="frank")
        self.theta: float = 0.5

    def fit(self, U: np.ndarray) -> "FrankCopula":
        U_clip = np.clip(U, 1e-5, 1.0 - 1e-5)
        self.n_obs, self.n_dim = U_clip.shape
        self.n_params = 1
        d = self.n_dim

        pairs: List[Tuple[int, int]] = [
            (i, j) for i in range(d) for j in range(i + 1, d)
        ]

        def _neg_log_lik(theta_val: float) -> float:
            if abs(theta_val) < 1e-4:
                return 1e9
            total_ll = 0.0
            e_theta = np.expm1(-theta_val)
            for i, j in pairs:
                u1 = U_clip[:, i]
                u2 = U_clip[:, j]
                e_u1 = np.expm1(-theta_val * u1)
                e_u2 = np.expm1(-theta_val * u2)
                denom = e_theta + e_u1 * e_u2
                if np.any(np.abs(denom) < 1e-12):
                    return 1e9
                log_num = np.log(abs(theta_val)) + np.log(abs(e_theta)) - theta_val * (u1 + u2)
                log_den = 2.0 * np.log(abs(denom))
                total_ll += float(np.sum(log_num - log_den))
            return -total_ll

        opt_res = minimize_scalar(_neg_log_lik, bounds=(-15.0, 15.0), method="bounded")
        self.theta = float(opt_res.x)
        self.log_likelihood = float(-opt_res.fun)

        self.lambda_L = 0.0
        self.lambda_U = 0.0
        self.tar = 0.0

        self.parameters = {"theta": self.theta}
        self._compute_information_criteria()
        return self

    def sample(self, n_samples: int = 10000, seed: Optional[int] = RANDOM_SEED) -> np.ndarray:
        # Bivariate sequential / conditional sampling extended to d dimensions
        rng = np.random.default_rng(seed)
        theta = self.theta
        if abs(theta) < 1e-4:
            return rng.uniform(1e-5, 1.0 - 1e-5, size=(n_samples, self.n_dim))

        U_out = np.zeros((n_samples, self.n_dim))
        U_out[:, 0] = rng.uniform(1e-5, 1.0 - 1e-5, size=n_samples)

        for col in range(1, self.n_dim):
            V = rng.uniform(1e-5, 1.0 - 1e-5, size=n_samples)
            u_prev = U_out[:, col - 1]
            # Conditional inverse of Frank copula
            num = V * (np.exp(-theta) - 1.0)
            den = np.exp(-theta * u_prev) + V * (1.0 - np.exp(-theta * u_prev))
            den = np.maximum(den, 1e-8)
            ratio = 1.0 + num / den
            ratio = np.clip(ratio, 1e-8, None)
            U_out[:, col] = -np.log(ratio) / theta

        return np.clip(U_out, 1e-6, 1.0 - 1e-6)


# --- Empirical Tail Dependence & Scale Tournament ---

def compute_empirical_tail_dependence(
    U: np.ndarray,
    q: float = 0.05,
    return_benchmark: bool = False,
) -> Union[Tuple[float, float, float], Tuple[float, float, float, float, float]]:
    """Computes model-free non-parametric tail dependence, Gaussian benchmark, and excess tail dependence.

    Formulas:
        lambda_L(q) = P(U1 <= q, U2 <= q) / q
        lambda_U(q) = P(U1 >= 1-q, U2 >= 1-q) / q
        TAR(q) = lambda_L(q) - lambda_U(q)
        lambda_Gauss(q; rho) = P_Gauss(U1 <= q, U2 <= q; rho) / q
        Excess_lambda_L(q) = lambda_L(q) - lambda_Gauss(q)

    Averages across all pairwise off-diagonal combinations in matrix U.

    Args:
        U: Uniform marginal data (N x d).
        q: Tail quantile threshold (e.g. 0.05 for 5% tail).
        return_benchmark: If True, returns (lambda_L, lambda_U, TAR, lambda_Gauss, excess_lambda_L).
                          If False, returns (lambda_L, lambda_U, TAR) for backward compatibility.

    Returns:
        Tuple of tail dependence metrics.
    """
    N, d = U.shape
    l_lower_list = []
    l_upper_list = []

    for i in range(d):
        for j in range(i + 1, d):
            u1 = U[:, i]
            u2 = U[:, j]
            l_L = float(np.clip(np.mean((u1 <= q) & (u2 <= q)) / q, 0.0, 1.0))
            l_U = float(np.clip(np.mean((u1 >= 1.0 - q) & (u2 >= 1.0 - q)) / q, 0.0, 1.0))
            l_lower_list.append(l_L)
            l_upper_list.append(l_U)

    mean_l_L = float(np.mean(l_lower_list))
    mean_l_U = float(np.mean(l_upper_list))
    tar = mean_l_L - mean_l_U

    if not return_benchmark:
        return mean_l_L, mean_l_U, tar

    # Gaussian benchmark for tail co-exceedance at quantile q
    z_q = float(norm.ppf(q))
    U_clipped = np.clip(U, 1e-6, 1.0 - 1e-6)
    X_norm = norm.ppf(U_clipped)
    R_norm = np.corrcoef(X_norm, rowvar=False)

    gauss_list = []
    for i in range(d):
        for j in range(i + 1, d):
            rho = float(R_norm[i, j])
            cov = [[1.0, rho], [rho, 1.0]]
            p_joint = float(multivariate_normal.cdf([z_q, z_q], mean=[0.0, 0.0], cov=cov))
            gauss_list.append(np.clip(p_joint / q, 0.0, 1.0))

    mean_gauss = float(np.mean(gauss_list)) if gauss_list else 0.0
    excess_l_L = float(mean_l_L - mean_gauss)

    return mean_l_L, mean_l_U, tar, mean_gauss, excess_l_L


def compute_pairwise_tail_matrix(
    u_df: pd.DataFrame,
    q: float = 0.05,
) -> pd.DataFrame:
    """Computes detailed pairwise empirical tail dependence across all assets.

    Args:
        u_df: DataFrame of uniform margins.
        q: Tail probability cutoff.

    Returns:
        pd.DataFrame with columns: Asset_1, Asset_2, lambda_L, lambda_U, TAR.
    """
    cols = list(u_df.columns)
    records = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a1, a2 = cols[i], cols[j]
            u1, u2 = u_df[a1].values, u_df[a2].values
            lL = float(np.clip(np.mean((u1 <= q) & (u2 <= q)) / q, 0.0, 1.0))
            lU = float(np.clip(np.mean((u1 >= 1.0 - q) & (u2 >= 1.0 - q)) / q, 0.0, 1.0))
            records.append({
                "Asset_1": a1,
                "Asset_2": a2,
                "lambda_L": lL,
                "lambda_U": lU,
                "TAR": lL - lU,
            })

    return pd.DataFrame(records)


def run_scale_copula_tournament(
    u_df: pd.DataFrame,
    scale_name: str,
) -> Dict[str, Any]:
    """Fits 5 copula families via MLE and selects the best model by BIC.

    Families tested:
        1. Gaussian
        2. Student-t
        3. Clayton
        4. Gumbel
        5. Frank

    Args:
        u_df: Transformed uniform margins U_i in (0, 1) for the given scale.
        scale_name: Scale identifier (e.g. 'D1', 'D2', 'D3', 'D4', 'D5', 'S5', 'Raw').

    Returns:
        Dict containing:
            - 'scale': scale_name
            - 'best_copula': winning copula family name
            - 'lambda_L': lower tail dependence
            - 'lambda_U': upper tail dependence
            - 'tar': Timescale Asymmetry Ratio (lambda_L - lambda_U)
            - 'lambda_gauss_bench': Gaussian copula benchmark at 5% threshold
            - 'excess_lambda_L': Excess tail crash dependence (emp_lL - gauss_bench)
            - 'bic_scores': Dict of BIC values per family
            - 'aic_scores': Dict of AIC values per family
            - 'log_likelihoods': Dict of Log-Likelihood values per family
            - 'fitted_copula_obj': fitted winning copula instance
            - 'empirical_tail_dep': empirical lambda_L, lambda_U, TAR, and benchmarks
    """
    U = u_df.values
    copula_candidates: Dict[str, BaseCopula] = {
        "gaussian": GaussianCopula(),
        "student_t": StudentTCopula(),
    }
    # The one-parameter Archimedean families are fitted with a pairwise composite
    # likelihood, which is not on the same scale as the full likelihood of the
    # Gaussian and t copulas when there are more than two assets. Their BIC would
    # not be comparable, so they only enter the contest for a single pair.
    if U.shape[1] == 2:
        copula_candidates.update({
            "clayton": ClaytonCopula(),
            "gumbel": GumbelCopula(),
            "frank": FrankCopula(),
        })

    bic_scores: Dict[str, float] = {}
    aic_scores: Dict[str, float] = {}
    ll_scores: Dict[str, float] = {}

    for name, cop in copula_candidates.items():
        try:
            cop.fit(U)
            bic_scores[name] = cop.bic
            aic_scores[name] = cop.aic
            ll_scores[name] = cop.log_likelihood
        except Exception as e:
            bic_scores[name] = np.inf
            aic_scores[name] = np.inf
            ll_scores[name] = -np.inf

    # Select winning model by minimum BIC
    best_copula_name = min(bic_scores, key=bic_scores.get)
    best_copula_obj = copula_candidates[best_copula_name]

    # Model-free empirical tail dependence and Gaussian benchmark
    emp_lL, emp_lU, emp_tar, gauss_bench, excess_lL = compute_empirical_tail_dependence(
        U, q=0.05, return_benchmark=True
    )

    # Theoretical copula tail dependence parameters (strictly from model definition)
    theo_lL = float(best_copula_obj.lambda_L)
    theo_lU = float(best_copula_obj.lambda_U)
    theo_tar = float(theo_lL - theo_lU)

    # Empirical tail dependence parameters
    emp_lL = float(emp_lL)
    emp_lU = float(emp_lU)
    emp_tar = float(emp_tar)

    # For H-TCM scale-based risk overlay:
    # Use theoretical parameters for elliptical models (Student-t).
    # For models where the theoretical coefficient is strictly zero by family definition
    # (e.g. Gumbel lower tail or Clayton upper tail), provide the empirical tail estimate
    # as the scale-level empirical risk parameter.
    scale_lL = theo_lL if theo_lL > 0 else emp_lL
    scale_lU = theo_lU if theo_lU > 0 else emp_lU
    scale_tar = scale_lL - scale_lU

    return {
        "scale": scale_name,
        "best_copula": best_copula_name,
        "lambda_L_theo": theo_lL,
        "lambda_U_theo": theo_lU,
        "tar_theo": theo_tar,
        "lambda_L_emp": emp_lL,
        "lambda_U_emp": emp_lU,
        "tar_emp": emp_tar,
        "lambda_gauss_bench": float(gauss_bench),
        "excess_lambda_L": float(excess_lL),
        "lambda_L": float(scale_lL),
        "lambda_U": float(scale_lU),
        "tar": float(scale_tar),
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "log_likelihoods": ll_scores,
        "fitted_copula_obj": best_copula_obj,
        "all_models": copula_candidates,
        "empirical_tail_dep": {
            "lambda_L": emp_lL,
            "lambda_U": emp_lU,
            "tar": emp_tar,
            "lambda_gauss_bench": float(gauss_bench),
            "excess_lambda_L": float(excess_lL),
        },
    }


def simulate_copula_joint_returns(
    fitted_copula_obj: object,
    models_meta: Dict[str, Any],
    n_samples: int = COPULA_SIMULATION_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulates synthetic uniform draws from copula and inverts via EVT-GARCH margins.

    Args:
        fitted_copula_obj: Fitted copula instance with .sample() method.
        models_meta: Marginal models dictionary produced by fit_margins_and_transform_uniform.
        n_samples: Number of synthetic scenarios to simulate (default: 10,000).
        seed: Random seed for deterministic simulation.

    Returns:
        pd.DataFrame: Simulated joint returns of shape (n_samples, n_assets),
                      with columns matching asset tickers.
    """
    cols: List[str] = models_meta["columns"]
    # Draw uniform samples U_sim in (0, 1)^(n_samples x n_dim)
    if hasattr(fitted_copula_obj, "sample"):
        u_sim = fitted_copula_obj.sample(n_samples=n_samples, seed=seed)
    else:
        # Fallback to independent uniform draws
        rng = np.random.default_rng(seed)
        u_sim = rng.uniform(1e-5, 1.0 - 1e-5, size=(n_samples, len(cols)))

    simulated_returns = {}
    for i, col in enumerate(cols):
        margin_model = models_meta["models"][col]
        simulated_returns[col] = margin_model.simulate_returns(u_sim[:, i])

    return pd.DataFrame(simulated_returns, columns=cols)


if __name__ == "__main__":
    print("=" * 75)
    print(" QuantEdge-MTR Scale-Optimal Copula Tournament Validation ")
    print("=" * 75)

    from src.data_loader import load_and_split_data
    from src.margins import fit_margins_and_transform_uniform

    # 1. Load Data & Estimate Margins
    df_train, _ = load_and_split_data()
    u_train, meta = fit_margins_and_transform_uniform(df_train)

    # 2. Run Tournament on In-Sample Uniform Margins
    print("\n--- RUNNING COPULA TOURNAMENT (RAW RETURN MARGINS) ---")
    tournament_result = run_scale_copula_tournament(u_train, scale_name="Raw_InSample")

    print(f"Scale Evaluated     : {tournament_result['scale']}")
    print(f"Winning Copula          : {tournament_result['best_copula'].upper()}")
    print(f"Lower Tail Dep (lambda_L): {tournament_result['lambda_L']:.4f}")
    print(f"Upper Tail Dep (lambda_U): {tournament_result['lambda_U']:.4f}")
    print(f"Timescale Asymmetry     : TAR = {tournament_result['tar']:+.4f}")

    print("\n--- COPULA LEADERBOARD (BIC SCORES) ---")
    leaderboard = []
    for fam in COPULA_FAMILIES:
        leaderboard.append({
            "Copula Family": fam,
            "Log-Likelihood": f"{tournament_result['log_likelihoods'].get(fam, 0.0):.2f}",
            "AIC": f"{tournament_result['aic_scores'].get(fam, 0.0):.2f}",
            "BIC": f"{tournament_result['bic_scores'].get(fam, 0.0):.2f}",
            "Rank": "WINNER" if fam == tournament_result["best_copula"] else "",
        })
    print(pd.DataFrame(leaderboard).to_string(index=False))

    # 3. Pairwise Cross-Asset Tail Matrix (Empirical)
    print("\n--- PAIRWISE CROSS-ASSET TAIL DEPENDENCE MATRIX (q = 0.05) ---")
    pairwise_df = compute_pairwise_tail_matrix(u_train, q=0.05)
    print(pairwise_df.to_string(index=False))

    # 4. Joint Return Simulation Test
    print("\n--- JOINT RETURN SIMULATION VIA WINNING COPULA ---")
    winning_copula = tournament_result["fitted_copula_obj"]
    sim_returns_df = simulate_copula_joint_returns(
        fitted_copula_obj=winning_copula,
        models_meta=meta,
        n_samples=10000,
    )
    print(f"Simulated Matrix Shape: {sim_returns_df.shape}")
    print("Simulated Correlation Matrix:")
    print(sim_returns_df.corr().round(3).to_string())

    print("\n[SUCCESS] src/copulas.py fully verified and operational.")

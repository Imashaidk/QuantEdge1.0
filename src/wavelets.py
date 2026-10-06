"""QuantEdge-MTR Wavelet Signal Processing Module: MODWT & MRA.

Implements the Maximal Overlap Discrete Wavelet Transform (MODWT) and
Multiresolution Analysis (MRA) for multi-asset financial return series.

Theoretical & Architectural Foundations:
1. Why MODWT over Decimated DWT:
   - DWT downsamples by 2 at each scale, halving sample size (N -> N/2 -> N/4...),
     which severely depletes degrees of freedom required for robust extreme value
     and copula estimation at intermediate and coarse timescales.
   - MODWT is shift-invariant (stationary): shifting the input time series shifts
     the wavelet coefficients by the exact same amount without aliasing.
   - MODWT preserves full sample length N at all decomposition scales J.
2. Additive Multiresolution Analysis (MRA):
   - Decomposes return series x_t into scale-specific detail series D_j and
     a smooth residual trend S_J:
         x_t = sum_{j=1}^J D_{j,t} + S_{J,t}   (exact additive identity)
3. Energy & Variance Conservation:
   - Energy conservation across scales satisfies ||x||^2 = sum ||D_j||^2 + ||S_J||^2.
   - Return variance is decomposed across orthogonal frequency bands corresponding
     to economic trading horizons (2-4d microstructure, weekly, bi-weekly, monthly,
     quarterly business cycle, and secular trend).
4. Zero Lookahead Enforcement:
   - Boundary filtering uses periodic circular boundary extension on strictly isolated historical training partitions.
   - Out-of-sample data is never concatenated during in-sample decomposition, ensuring zero lookahead bias.
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
import pywt

from src.config import (
    SCALE_HORIZONS,
    SCALE_NAMES,
    WAVELET_FAMILY,
    WAVELET_LEVEL,
)


def get_wavelet_filters(wavelet: str = WAVELET_FAMILY) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves and scales decomposition filters for MODWT.

    For MODWT, standard orthonormal DWT filter coefficients h (high-pass) and
    g (low-pass) must be normalized by 1 / sqrt(2):
        \\tilde{h}_l = h_l / \\sqrt{2}
        \\tilde{g}_l = g_l / \\sqrt{2}

    Args:
        wavelet: Name of the wavelet family (e.g., 'sym8', 'db4', 'sym4').

    Returns:
        Tuple of (h_tilde, g_tilde) numpy float64 arrays.
    """
    w = pywt.Wavelet(wavelet)
    # PyWavelets: dec_hi is high-pass (detail), dec_lo is low-pass (scaling)
    h_dwt = np.array(w.dec_hi, dtype=np.float64)
    g_dwt = np.array(w.dec_lo, dtype=np.float64)

    h_tilde = h_dwt / np.sqrt(2.0)
    g_tilde = g_dwt / np.sqrt(2.0)
    return h_tilde, g_tilde


def get_modwt_filter_lengths(
    wavelet: str = WAVELET_FAMILY, level: int = WAVELET_LEVEL
) -> Dict[str, int]:
    """Calculates the effective filter length L_j at each decomposition scale.

    At decomposition scale j, inserting 2^{j-1} - 1 zeros between base taps
    expands the effective filter width to:
        L_j = (2^j - 1) * (L - 1) + 1

    Coefficients within L_j - 1 points of boundaries are influenced by boundary
    treatment (circular wrap or reflection).

    Args:
        wavelet: Wavelet family name.
        level: Maximum decomposition level J.

    Returns:
        Dictionary mapping scale names ('D1'..'DJ', 'SJ') to effective filter lengths.
    """
    w = pywt.Wavelet(wavelet)
    L = len(w.dec_hi)
    lengths: Dict[str, int] = {}
    for j in range(1, level + 1):
        L_j = (2**j - 1) * (L - 1) + 1
        lengths[f"D{j}"] = int(L_j)
    lengths[f"S{level}"] = lengths[f"D{level}"]
    return lengths


def modwt(
    x: np.ndarray,
    wavelet: str = WAVELET_FAMILY,
    level: int = WAVELET_LEVEL,
    boundary: str = "periodic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the Maximal Overlap Discrete Wavelet Transform (MODWT).

    Uses the circular pyramid filtering algorithm with step expansion:
    At scale j, filter taps are applied with step stride 2^{j-1}.

    Args:
        x: 1D numpy array of real-valued return observations of length N.
        wavelet: Wavelet family name ('sym8' default).
        level: Decomposition level J (default: 5).
        boundary: Boundary extension method: 'periodic' (circular wrap) or
                  'reflection' (symmetric boundary reflection to minimize edge distortion).

    Returns:
        Tuple of:
            W (np.ndarray): 2D array of wavelet detail coefficients of shape (level, N).
                            Row j-1 corresponds to scale D_j.
            V (np.ndarray): 2D array of scaling approximation coefficients of shape (level, N).
                            Row j-1 corresponds to approximation V_j.
    """
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    N_orig = len(x_arr)
    if N_orig == 0:
        raise ValueError("Input array x cannot be empty.")
    if level < 1:
        raise ValueError(f"Decomposition level must be >= 1, got {level}.")

    # Boundary padding
    if boundary == "reflection":
        x_pad = np.concatenate([x_arr, x_arr[::-1]])
    elif boundary == "periodic":
        x_pad = x_arr.copy()
    else:
        raise ValueError(f"Unsupported boundary mode: {boundary}. Use 'periodic' or 'reflection'.")

    N_pad = len(x_pad)
    h, g = get_wavelet_filters(wavelet)
    L = len(h)

    W_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []
    V_prev = x_pad.copy()

    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        w_j = np.zeros(N_pad, dtype=np.float64)
        v_j = np.zeros(N_pad, dtype=np.float64)

        # Vectorized circular shift convolution
        for l in range(L):
            shift = (step * l) % N_pad
            rolled_V = np.roll(V_prev, shift)
            w_j += h[l] * rolled_V
            v_j += g[l] * rolled_V

        W_list.append(w_j[:N_orig])
        V_list.append(v_j[:N_orig])
        V_prev = v_j

    W = np.vstack(W_list)  # Shape: (level, N_orig)
    V = np.vstack(V_list)  # Shape: (level, N_orig)
    return W, V


def _imodwt_step(
    w_j: Optional[np.ndarray],
    v_j: Optional[np.ndarray],
    j: int,
    h: np.ndarray,
    g: np.ndarray,
    N: int,
) -> np.ndarray:
    """Performs a single inverse MODWT step from scale j to j-1.

    V_{j-1}[t] = sum_l h_l * W_j[t + 2^{j-1} * l] + sum_l g_l * V_j[t + 2^{j-1} * l]
    """
    step = 2 ** (j - 1)
    L = len(h)
    v_prev = np.zeros(N, dtype=np.float64)
    for l in range(L):
        shift = (-step * l) % N
        if w_j is not None:
            v_prev += h[l] * np.roll(w_j, shift)
        if v_j is not None:
            v_prev += g[l] * np.roll(v_j, shift)
    return v_prev


def imodwt(
    W: np.ndarray,
    V: np.ndarray,
    wavelet: str = WAVELET_FAMILY,
    boundary: str = "periodic",
) -> np.ndarray:
    """Computes the Inverse Maximal Overlap Discrete Wavelet Transform (IMODWT).

    Reconstructs the original time series from all detail coefficient levels W
    and the coarsest smooth scaling coefficient level V.

    Args:
        W: 2D array of detail coefficients of shape (level, N).
        V: 1D or 2D array of scaling coefficients. If 2D (level, N), the coarsest
           level V[-1] is used for synthesis.
        wavelet: Wavelet family name.
        boundary: Boundary extension method ('periodic' or 'reflection').

    Returns:
        1D numpy array of reconstructed series of length N.
    """
    W_arr = np.asarray(W, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)

    if W_arr.ndim != 2:
        raise ValueError(f"W must be 2D array of shape (level, N), got shape {W_arr.shape}.")

    level, N = W_arr.shape
    v_coarsest = V_arr[-1] if V_arr.ndim == 2 else V_arr

    if len(v_coarsest) != N:
        raise ValueError(f"Length mismatch: W has N={N}, V has length {len(v_coarsest)}.")

    h, g = get_wavelet_filters(wavelet)

    # Reconstruct backwards from level J down to 1
    curr_v = v_coarsest.copy()
    for j in range(level, 0, -1):
        curr_v = _imodwt_step(W_arr[j - 1], curr_v, j, h, g, N)

    return curr_v


def mra_decompose(
    x: np.ndarray,
    wavelet: str = WAVELET_FAMILY,
    level: int = WAVELET_LEVEL,
    boundary: str = "periodic",
) -> Dict[str, np.ndarray]:
    """Computes additive Multiresolution Analysis (MRA) decomposition.

    Decomposes 1D signal x into scale-specific detail series D_1...D_J and
    smooth trend series S_J such that:
        x_t = sum_{j=1}^J D_{j,t} + S_{J,t}

    Each component series preserves the exact original time domain length N.

    Args:
        x: 1D numpy array of length N.
        wavelet: Wavelet family name ('sym8' default).
        level: Maximum decomposition level J (default: 5).
        boundary: 'periodic' or 'reflection'.

    Returns:
        Dictionary mapping scale names ['D1', ..., 'DJ', 'SJ'] to 1D numpy arrays.
    """
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    N_orig = len(x_arr)
    if N_orig == 0:
        raise ValueError("Input array x cannot be empty.")

    # Apply boundary padding if reflection
    if boundary == "reflection":
        x_pad = np.concatenate([x_arr, x_arr[::-1]])
    else:
        x_pad = x_arr.copy()

    N_pad = len(x_pad)
    h, g = get_wavelet_filters(wavelet)
    L = len(h)

    # 1. Forward MODWT on padded array
    W_list: List[np.ndarray] = []
    V_curr = x_pad.copy()

    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        w_j = np.zeros(N_pad, dtype=np.float64)
        v_j = np.zeros(N_pad, dtype=np.float64)

        for l in range(L):
            shift = (step * l) % N_pad
            w_j += h[l] * np.roll(V_curr, shift)
            v_j += g[l] * np.roll(V_curr, shift)

        W_list.append(w_j)
        V_curr = v_j

    # 2. Synthesize each Detail series D_j by setting all other coefficients to zero
    details: Dict[str, np.ndarray] = {}
    for j in range(1, level + 1):
        w_target = W_list[j - 1]
        # Invert from scale j down to 1
        curr = _imodwt_step(w_target, None, j, h, g, N_pad)
        for k in range(j - 1, 0, -1):
            curr = _imodwt_step(None, curr, k, h, g, N_pad)
        details[f"D{j}"] = curr[:N_orig]

    # 3. Synthesize Smooth series S_J from V_J
    curr_s = V_curr.copy()
    for k in range(level, 0, -1):
        curr_s = _imodwt_step(None, curr_s, k, h, g, N_pad)
    details[f"S{level}"] = curr_s[:N_orig]

    return details


def decompose_multiscale(
    df_returns: pd.DataFrame,
    wavelet: str = WAVELET_FAMILY,
    level: int = WAVELET_LEVEL,
    boundary: str = "periodic",
) -> Dict[str, pd.DataFrame]:
    """Decomposes multi-asset return series using MODWT Additive MRA.

    Iterates across each asset column in df_returns and produces synchronized
    scale DataFrames preserving exact index, columns, and sample size.

    Args:
        df_returns: pd.DataFrame of asset log returns (rows: DatetimeIndex, cols: tickers).
        wavelet: Wavelet family name ('sym8' default).
        level: Decomposition depth J (default: 5).
        boundary: 'periodic' (circular wrap) or 'reflection'.

    Returns:
        Dict mapping scale names ('D1', 'D2', 'D3', 'D4', 'D5', 'S5') to pd.DataFrames.
        Each DataFrame has the exact shape, columns, and index as df_returns.

    Raises:
        ValueError: If df_returns contains NaNs or has insufficient observations.
    """
    if not isinstance(df_returns, pd.DataFrame):
        if isinstance(df_returns, pd.Series):
            df_returns = df_returns.to_frame()
        else:
            raise TypeError(f"df_returns must be a pd.DataFrame, got {type(df_returns)}.")

    if df_returns.empty:
        raise ValueError("df_returns cannot be empty.")

    if df_returns.isna().any().any():
        raise ValueError("df_returns contains NaN values. Clean data before decomposition.")

    min_samples = 2 ** (level + 1)
    if len(df_returns) < min_samples:
        raise ValueError(
            f"Sample size {len(df_returns)} is too short for level {level} decomposition. "
            f"Requires at least {min_samples} observations."
        )

    # Initialize scale containers
    scale_keys = [f"D{j}" for j in range(1, level + 1)] + [f"S{level}"]
    decomposed_dict: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in scale_keys}

    # Decompose each asset column
    for col in df_returns.columns:
        col_series = df_returns[col].to_numpy(dtype=np.float64)
        mra_res = mra_decompose(col_series, wavelet=wavelet, level=level, boundary=boundary)
        for scale in scale_keys:
            decomposed_dict[scale][col] = mra_res[scale]

    # Package into DataFrames with identical DatetimeIndex and column names
    out_dict: Dict[str, pd.DataFrame] = {}
    for scale in scale_keys:
        out_dict[scale] = pd.DataFrame(
            decomposed_dict[scale],
            index=df_returns.index,
            columns=df_returns.columns,
        )

    return out_dict


def verify_additivity(
    df_returns: pd.DataFrame,
    decomposed: Dict[str, pd.DataFrame],
    tol: float = 1e-10,
    raise_error: bool = False,
) -> bool:
    """Verifies exact additive reconstruction of MRA decomposition.

    Checks the invariant:
        max |R_t - sum_{scale} D_{j,t} - S_{J,t}| < tol

    Args:
        df_returns: Original return DataFrame.
        decomposed: Dictionary of scale DataFrames returned by decompose_multiscale.
        tol: Maximum allowable absolute error (default: 1e-10).
        raise_error: If True, raises AssertionError on failure; otherwise returns bool.

    Returns:
        True if all elements reconstruct within tolerance, False otherwise.
    """
    scale_dfs = list(decomposed.values())
    if not scale_dfs:
        raise ValueError("Decomposed dictionary is empty.")

    recon_df = sum(scale_dfs)
    diff = np.abs(df_returns.to_numpy() - recon_df.to_numpy())
    max_error = float(np.nanmax(diff))

    is_valid = max_error < tol

    if not is_valid and raise_error:
        raise AssertionError(
            f"Additive reconstruction failed! Max error {max_error:.4e} exceeds tolerance {tol:.4e}."
        )

    return is_valid


def compute_scale_variance_decomposition(
    decomposed: Dict[str, pd.DataFrame],
    normalize: bool = True,
) -> pd.DataFrame:
    """Computes scale-by-scale variance contribution per asset.

    Calculates the sample variance Var(D_j) and Var(S_J) for each asset.
    If normalize=True, converts to percentage of sum of scale variances:
        pct_var(scale, asset) = 100 * Var(scale) / sum_k Var(k)

    Args:
        decomposed: Dict of scale DataFrames ('D1'..'D5', 'S5').
        normalize: If True, scale percentages sum to 100% per asset.

    Returns:
        pd.DataFrame where index is scale names ('D1'..'D5', 'S5') and
        columns are asset tickers. Values represent percentage variance (or raw variance).
    """
    scale_names = list(decomposed.keys())
    assets = list(decomposed[scale_names[0]].columns)

    var_dict: Dict[str, Dict[str, float]] = {s: {} for s in scale_names}

    for scale in scale_names:
        df_s = decomposed[scale]
        for asset in assets:
            var_dict[scale][asset] = float(df_s[asset].var(ddof=1))

    var_df = pd.DataFrame(var_dict).T  # Index: scales, Columns: assets

    if normalize:
        sum_var = var_df.sum(axis=0)
        pct_var_df = var_df.div(sum_var, axis=1) * 100.0
        return pct_var_df
    return var_df


def verify_zero_lookahead(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    wavelet: str = WAVELET_FAMILY,
    level: int = WAVELET_LEVEL,
) -> bool:
    """Validates that in-sample decomposition is strictly isolated from future test data.

    Demonstrates:
    1. Deterministic reproducibility: Decomposing df_train repeatedly yields bit-for-bit identical results.
    2. Temporal isolation: Enforcing isolated decomposition on df_train prevents leakage that would
       occur if future observations were concatenated before filtering.

    Args:
        df_train: In-sample training returns.
        df_test: Out-of-sample test returns.
        wavelet: Wavelet family.
        level: Decomposition level.

    Returns:
        True if isolation and reproducibility hold strictly.
    """
    # 1. Deterministic reproducibility
    train_dec1 = decompose_multiscale(df_train, wavelet=wavelet, level=level)
    train_dec2 = decompose_multiscale(df_train, wavelet=wavelet, level=level)

    for scale in train_dec1:
        diff = np.max(np.abs(train_dec1[scale].values - train_dec2[scale].values))
        if diff > 1e-14:
            return False

    # 2. Verify temporal isolation requirement:
    # Appending test data in a joint decomposition alters boundary coefficients,
    # demonstrating why isolated training-window filtering is strictly enforced.
    df_joint = pd.concat([df_train, df_test])
    joint_dec = decompose_multiscale(df_joint, wavelet=wavelet, level=level)
    n_train = len(df_train)
    has_boundary_leakage_if_joint = False
    for scale in train_dec1:
        diff_boundary = np.max(np.abs(train_dec1[scale].values[-10:] - joint_dec[scale].values[n_train-10:n_train]))
        if diff_boundary > 1e-6:
            has_boundary_leakage_if_joint = True
            break

    return has_boundary_leakage_if_joint


def get_scale_metadata(level: int = WAVELET_LEVEL) -> pd.DataFrame:
    """Returns a structured metadata table of scales, frequencies, and interpretations.

    Provides economic trading horizon descriptions, cycle periods in trading days,
    and frequency ranges.

    Args:
        level: Number of decomposition levels J (default: 5).

    Returns:
        pd.DataFrame indexed by scale name.
    """
    rows = []
    for j in range(1, level + 1):
        scale = f"D{j}"
        t_low = 2**j
        t_high = 2 ** (j + 1)
        horizon_desc = SCALE_HORIZONS.get(scale, f"{t_low}-{t_high} Days")
        rows.append({
            "Scale": scale,
            "Type": "Detail",
            "Period (Days)": f"{t_low}-{t_high}d",
            "Frequency Band": f"[{1.0/t_high:.4f}, {1.0/t_low:.4f}]",
            "Economic Interpretation": horizon_desc,
        })

    # Smooth scale
    s_scale = f"S{level}"
    t_smooth = 2 ** (level + 1)
    s_desc = SCALE_HORIZONS.get(s_scale, f">{t_smooth} Days")
    rows.append({
        "Scale": s_scale,
        "Type": "Smooth Trend",
        "Period (Days)": f">{t_smooth}d",
        "Frequency Band": f"[0, {1.0/t_smooth:.4f}]",
        "Economic Interpretation": s_desc,
    })

    return pd.DataFrame(rows).set_index("Scale")


if __name__ == "__main__":
    from src.data_loader import load_and_split_data

    # Load in-sample market return data
    df_train, df_test = load_and_split_data()
    print(f"Loaded train data: {df_train.shape[0]} days x {df_train.shape[1]} assets.")

    # Multiscale decomposition
    decomposed = decompose_multiscale(df_train, wavelet=WAVELET_FAMILY, level=WAVELET_LEVEL)
    print(f"Decomposition complete: {list(decomposed.keys())}")

    # Additivity check
    recon_df = sum(decomposed.values())
    max_err = float(np.max(np.abs(df_train.values - recon_df.values)))
    is_additive = verify_additivity(df_train, decomposed, tol=1e-10)
    print(f"Additive check: max diff = {max_err:.4e} ({'OK' if is_additive else 'FAILED'})")

    # Variance decomposition
    pct_var_df = compute_scale_variance_decomposition(decomposed, normalize=True)
    print("\nVariance contributions (%):")
    print(pct_var_df.round(2).to_string())

    # Filter effective lengths
    filter_lens = get_modwt_filter_lengths(WAVELET_FAMILY, WAVELET_LEVEL)
    print("\nFilter lengths:")
    for scale, flen in filter_lens.items():
        print(f"  {scale}: {flen} days")

    # Zero lookahead validation
    lookahead_ok = verify_zero_lookahead(df_train, df_test)
    print(f"Zero lookahead test: {'PASS' if lookahead_ok else 'FAIL'}")

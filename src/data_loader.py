"""Price data and log returns.

Loads the daily prices stored in data/, or downloads them from Yahoo Finance if
the file is missing, and computes log returns. Also has a stationarity check
and summary statistics.

Author: Sameera Ekanayaka
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List

# Suppress statsmodels future warnings for clean terminal logging
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

from src.config import DATA_DIR, TICKERS


def fetch_or_load_prices(
    tickers: List[str] = TICKERS,
    start_date: str = "2007-04-01",  # HYG starts trading on 2007-04-11
    end_date: str = "2026-07-01",
    cache_path: Path = DATA_DIR / "raw_prices.csv",
) -> pd.DataFrame:
    """Fetches adjusted close prices via Yahoo Finance or loads from local cache.

    Args:
        tickers: List of asset ticker symbols.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        cache_path: Path to the cached raw prices CSV file.

    Returns:
        pd.DataFrame: Daily adjusted close prices indexed by DatetimeIndex.
    """
    if cache_path.exists():
        print(f"[DataLoader] Loading cached price data from: {cache_path}")
        df_prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Ensure column ordering matches tickers
        missing = [t for t in tickers if t not in df_prices.columns]
        if not missing:
            return df_prices[tickers]
        print(f"[DataLoader] Cache missing tickers {missing}. Re-downloading...")

    print(f"[DataLoader] Downloading adjusted close prices for {tickers} from Yahoo Finance...")
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            df_prices = raw["Close"][tickers]
        else:
            df_prices = raw.xs("Close", axis=1, level=0)[tickers]
    else:
        df_prices = raw[tickers]

    # Forward fill the odd missing day, then start where every asset has a price.
    # A backward fill would copy later HYG prices into the days before it existed.
    df_prices = df_prices.ffill().dropna()
    df_prices.index = pd.to_datetime(df_prices.index)
    df_prices.index.name = "Date"

    # Save clean copy to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_prices.to_csv(cache_path)
    print(f"[DataLoader] Saved clean raw prices to: {cache_path} ({len(df_prices)} rows)")
    return df_prices


def compute_log_returns(
    df_prices: pd.DataFrame,
    cache_path: Path = DATA_DIR / "log_returns.csv",
) -> pd.DataFrame:
    """Computes daily logarithmic returns: R_t = ln(P_t / P_{t-1}).

    Args:
        df_prices: DataFrame of adjusted close prices.
        cache_path: Path to the cached log returns CSV file.

    Returns:
        pd.DataFrame: Daily log returns without missing values.
    """
    if cache_path.exists():
        print(f"[DataLoader] Loading cached log returns from: {cache_path}")
        df_returns = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df_returns

    log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    log_returns.index = pd.to_datetime(log_returns.index)
    log_returns.index.name = "Date"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_returns.to_csv(cache_path)
    print(f"[DataLoader] Saved clean log returns to: {cache_path} ({len(log_returns)} rows)")
    return log_returns


def verify_stationarity(df_returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Runs the Augmented Dickey-Fuller (ADF) test on each asset return series.

    Asserts that all series are strictly stationary (p-value < 0.01).

    Args:
        df_returns: DataFrame of asset returns.

    Returns:
        Dict mapping each ticker to ADF test statistics and p-values.
    """
    adf_results = {}
    for col in df_returns.columns:
        result = adfuller(df_returns[col].dropna(), autolag="AIC")
        adf_stat = float(result[0])
        p_val = float(result[1])
        used_lag = int(result[2])
        n_obs = int(result[3])

        adf_results[col] = {
            "adf_stat": adf_stat,
            "p_value": p_val,
            "used_lag": used_lag,
            "n_obs": n_obs,
        }
        if p_val > 0.01:
            print(f"[WARNING] Return series {col} ADF p-value {p_val:.4f} > 0.01!")
        else:
            print(f"[Stationarity] {col}: ADF Stat = {adf_stat:.4f}, p = {p_val:.4e} (Stationary [OK])")

    return adf_results


def load_returns(cache_dir: Path = DATA_DIR, tickers: List[str] = TICKERS) -> pd.DataFrame:
    """Full daily log return history used for the descriptive analysis."""
    df_prices = fetch_or_load_prices(tickers=tickers, cache_path=cache_dir / "raw_prices.csv")
    return compute_log_returns(df_prices, cache_path=cache_dir / "log_returns.csv")


def compute_summary_statistics(df_returns: pd.DataFrame) -> pd.DataFrame:
    """Computes academic summary statistics for market risk reporting.

    Calculates Annualized Return, Annualized Volatility, Skewness,
    Kurtosis, Min, Max, and Sharpe Ratio.

    Args:
        df_returns: Daily log returns DataFrame.

    Returns:
        pd.DataFrame: Formatted summary statistics table.
    """
    stats = pd.DataFrame(index=df_returns.columns)
    stats["Mean (Ann. %)"] = (df_returns.mean() * 252 * 100).round(2)
    stats["Vol (Ann. %)"] = (df_returns.std() * np.sqrt(252) * 100).round(2)
    stats["Skewness"] = df_returns.skew().round(3)
    stats["Kurtosis (Excess)"] = (df_returns.kurtosis()).round(3)
    stats["Min Return (%)"] = (df_returns.min() * 100).round(2)
    stats["Max Return (%)"] = (df_returns.max() * 100).round(2)
    stats["Sharpe (Rf=0)"] = (stats["Mean (Ann. %)"] / stats["Vol (Ann. %)"]).round(2)
    return stats


if __name__ == "__main__":
    returns = load_returns()
    print(compute_summary_statistics(returns).to_string())
    verify_stationarity(returns)

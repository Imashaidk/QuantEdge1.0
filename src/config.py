"""Settings used across the project.

Asset universe, portfolio weights, paths, the random seed and every model
setting live here, so a change in one place reaches the whole pipeline.

Author: Sameera Ekanayaka
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Directory paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
FIGURES_DIR: Path = ROOT_DIR / "figures"
TABLES_DIR: Path = ROOT_DIR / "tables"
REPORT_DIR: Path = ROOT_DIR / "report"
RESULTS_DIR: Path = ROOT_DIR / "results"

# Ensure output directories exist
for directory in [DATA_DIR, FIGURES_DIR, TABLES_DIR, REPORT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Deterministic execution
RANDOM_SEED: int = 42

# Asset universe and allocations
TICKERS: List[str] = ["SPY", "QQQ", "TLT", "GLD", "HYG"]

ASSET_DESCRIPTIONS: Dict[str, str] = {
    "SPY": "S&P 500 ETF (Core Equity Beta)",
    "QQQ": "Nasdaq 100 ETF (High-Beta Growth / Tech)",
    "TLT": "20+ Year Treasury Bond ETF (Duration / Safe Haven)",
    "GLD": "SPDR Gold Shares (Inflation / Store of Value)",
    "HYG": "High Yield Corporate Bond ETF (Credit & Liquidity)",
}

# Baseline fixed diversified portfolio weights: [SPY, QQQ, TLT, GLD, HYG]
DEFAULT_PORTFOLIO_WEIGHTS: np.ndarray = np.array([0.30, 0.20, 0.25, 0.15, 0.10])

# Wavelet multiresolution analysis (MODWT)
WAVELET_FAMILY: str = "sym8"
WAVELET_LEVEL: int = 5
# Reflection avoids wrapping the last days of the sample onto the first ones,
# which distorts the coarse scales (the level 5 sym8 filter spans 466 days).
WAVELET_BOUNDARY: str = "reflection"
SCALE_NAMES: List[str] = ["D1", "D2", "D3", "D4", "D5", "S5"]

SCALE_HORIZONS: Dict[str, str] = {
    "D1": "2 to 4 days",
    "D2": "4 to 8 days",
    "D3": "8 to 16 days",
    "D4": "16 to 32 days",
    "D5": "32 to 64 days",
    "S5": "longer than 64 days",
}

# Econometric margins and copulas
EVT_TAIL_PERCENTILE: float = 0.10  # 10% upper and lower thresholds for GPD
COPULA_FAMILIES: List[str] = [
    "gaussian",
    "student_t",
    "clayton",
    "gumbel",
    "frank",
]
COPULA_SIMULATION_SAMPLES: int = 10_000

# Horizon-aware VaR. An h-day return keeps moves longer than about h days, so each
# horizon is matched to the low-pass wavelet view that keeps moves longer than
# 2^(j+1) days: 5d -> view 1 (> 4d), 20d -> view 3 (> 16d), 60d -> view 5 (> 64d).
HORIZON_VIEW: Dict[int, int] = {1: 0, 5: 1, 20: 3, 60: 5}
VAR_SIMULATIONS: int = 20_000
# Estimation window for every forecast, about four years of trading days.
ROLLING_WINDOW: int = 1000
# Re-estimate every model once a month; volatilities still update daily in between.
REFIT_EVERY: int = 21

# Tail co-exceedance is measured in the worst 5% of days for each asset.
TAIL_QUANTILE: float = 0.05
TAIL_BOOTSTRAP_REPS: int = 500

# Risk and regulatory backtest parameters
ALPHA_VAR_99: float = 0.99

# Evaluation horizons in days
BACKTEST_HORIZONS: List[int] = [1, 5, 20]

# Breaches are also counted inside these periods. The first forecast needs
# ROLLING_WINDOW days of history, so with data from April 2007 the backtest
# starts in 2011 and 2008 is only ever in-sample.
STRESS_PERIODS: Dict[str, Tuple[str, str]] = {
    "2011 US downgrade": ("2011-07-22", "2011-10-31"),
    "COVID 2020": ("2020-02-15", "2020-04-30"),
    "2022 rates shock": ("2022-01-01", "2022-10-31"),
    "2025 tariff shock": ("2025-03-25", "2025-05-30"),
}

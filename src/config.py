"""QuantEdge-MTR Configuration & Hyperparameters Module.

Defines global asset universe, sample date windows, random seeds,
path constants, wavelet levels, and risk model parameters.

Author: Sameera Ekanayaka
"""

from pathlib import Path
from typing import Dict, List
import numpy as np

# Directory paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
FIGURES_DIR: Path = ROOT_DIR / "figures"
TABLES_DIR: Path = ROOT_DIR / "tables"
REPORT_DIR: Path = ROOT_DIR / "report"

# Ensure output directories exist
for directory in [DATA_DIR, FIGURES_DIR, TABLES_DIR, REPORT_DIR]:
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

# Temporal partitioning
TRAIN_START: str = "2015-01-01"
TRAIN_END: str = "2022-12-31"

TEST_START: str = "2023-01-01"
TEST_END: str = "2026-06-30"

# Wavelet multiresolution analysis (MODWT)
WAVELET_FAMILY: str = "sym8"
WAVELET_LEVEL: int = 5
# Reflection avoids wrapping the last days of the sample onto the first ones,
# which distorts the coarse scales (the level 5 sym8 filter spans 466 days).
WAVELET_BOUNDARY: str = "reflection"
SCALE_NAMES: List[str] = ["D1", "D2", "D3", "D4", "D5", "S5"]

SCALE_HORIZONS: Dict[str, str] = {
    "D1": "2 to 4 Days (Microstructure Noise / Daily Rebalancing)",
    "D2": "4 to 8 Days (Weekly Swing / Momentum)",
    "D3": "8 to 16 Days (Bi-weekly Sentiment)",
    "D4": "16 to 32 Days (Monthly Portfolio Rebalancing)",
    "D5": "32 to 64 Days (Quarterly Business / Earnings Cycle)",
    "S5": ">64 Days (Macroeconomic Secular Trend)",
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

# Risk and regulatory backtest parameters
ALPHA_VAR_99: float = 0.99
ALPHA_VAR_95: float = 0.95
ALPHA_ES_99: float = 0.99
ALPHA_ES_975: float = 0.99  # Standardized alias matching alpha=0.99

# Evaluation horizons in days
BACKTEST_HORIZONS: List[int] = [1, 5, 20]

# Horizon-Conditioned Tail Capital Multiplier (H-TCM) calibration factor
HTCM_KAPPA: float = 0.35
HTCM_EPSILON: float = 1e-6

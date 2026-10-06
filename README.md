# QuantEdge 1.0: Risk Across Tails and Timescales

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Competition](https://img.shields.io/badge/SAIFA-Quant%20Edge%201.0-orange)](https://saifa.lk)

Official repository for **SAIFA Quant Edge 1.0 (Round 1: Initial Screening Challenge)**.  
**Submission Deadline:** Wednesday, 7 October 2026: 23:59 (Sri Lanka Time)

***

## The Core Research Question
> **"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"**

Standard risk models assume a single horizon and static dependence, scaling 1-day risk via $\sqrt{h}$. In real financial markets, assets co-move differently over days than over months, and joint crash dependence spikes during market turmoil. 

This project constructs the **QuantEdge-MTR (Multiscale Tail Risk Framework)**, combining:
1. **Wavelet Multiresolution Analysis (MODWT):** Decomposes multi-asset returns into discrete timescales (2 days to $>64$ days) with machine-precision additivity ($3.77 \times 10^{-14}$).
2. **Semi-Parametric ARMA-GARCH + EVT Margins:** Eliminates heteroskedasticity and accurately models extreme heavy tails via Peaks-Over-Threshold (POT) Generalized Pareto Distributions (GPD).
3. **Multiscale Asymmetric Copulas:** Runs an automated tournament across 5 families (Gaussian, Student-$t$, Clayton, Gumbel, Frank) to track the evolution of lower tail dependence $\lambda_L(h)$ across investment horizons.
4. **Out-of-Sample Backtesting (875 Days):** Benchmarks against conventional models (Historical Simulation, Parametric Gaussian, and Basel $\sqrt{h}$ scaling) using Kupiec POF, Christoffersen independence, Basel Traffic Light zones, and Fissler-Ziegel (FZ) loss.
5. **Actionable Institutional Policy:** The **Horizon-Conditioned Tail Capital Multiplier (H-TCM)** formula and sensitivity matrix.

***

## Core Empirical Findings

1. **Flight-to-Liquidity Contagion Paradox:** Lower tail crash dependence ($\lambda_L$) jumps by over **650%**, rising from $0.042$ at short noise horizons ($D_1$: 2 to 4 days) to $0.318$ at quarterly business cycles ($D_5$: 32 to 64 days).
2. **Crash Asymmetry:** The Timescale Asymmetry Ratio ($\text{TAR} = \lambda_L - \lambda_U$) surges from $0.000$ to $+0.286$. Assets crash together during extended sell-offs, but recover on their own.
3. **The H-TCM Rule:** 
   $$\text{VaR}_h^* = \text{VaR}_1 \times \sqrt{h} \times \left[ 1 + \kappa \cdot \max\left(0, \frac{\lambda_L(h) - \lambda_L(1)}{\lambda_L(1) + \epsilon}\right) \right]$$
   With $\kappa = 0.35$, it adds an 8.8% capital buffer at weekly horizons and a 24.5% buffer at monthly horizons, protecting against liquidity freezes.

***

## Repository Architecture

```text
QuantEdge/
├── data/                  # Cleaned historical asset price & return CSVs (SPY, QQQ, TLT, GLD, HYG)
├── docs/                  # Competition compliance & AI disclosure log
├── figures/               # Publication-grade figures (300 DPI) for final report
├── report/                # Academic paper (report.tex & REPORT.md)
│   ├── report.tex         # 2-column LaTeX submission paper
│   └── REPORT.md          # Complete Markdown companion
├── src/
│   ├── __init__.py
│   ├── config.py          # Portfolio tickers, date windows, hyperparameters, global seed
│   ├── data_loader.py     # Public data ingestion & log return calculation
│   ├── wavelets.py        # MODWT multiresolution decomposition
│   ├── margins.py         # ARMA-GARCH + EVT tail modeling
│   ├── copulas.py         # Copula tournament & tail dependence extraction
│   ├── risk_engine.py     # VaR / ES multiscale calculation & H-TCM
│   ├── backtest.py        # Kupiec, Christoffersen, Basel Traffic Light
│   └── visualizer.py      # Publication-grade plotting pipeline
├── tables/                # Auto-generated LaTeX tables
├── tests/                 # Automated unit tests (41/41 passing)
├── run_all.py             # Single-command runner reproducing all report figures/tables
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

***

## Team Work Breakdown Matrix

| Role | Lead | Target Modules | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Member 1** | Team Lead & Pipeline Architect | `src/config.py`<br>`src/data_loader.py`<br>`run_all.py` | Data fetching/caching, in/out-of-sample splitting, master runner, repo PR review, ZIP packaging. |
| **Member 2** | Wavelet & Signal Specialist | `src/wavelets.py`<br>`tests/test_wavelets.py` | MODWT decomposition ($D_1$ to $D_5, S_5$), additive reconstruction tests, zero-lookahead boundary filtering. |
| **Member 3** | Econometrician & Copula Modeler | `src/margins.py`<br>`src/copulas.py` | ARMA-GARCH + EVT-POT margins, uniform PIT validation, copula MLE fitting, tail dependence curves $\lambda_L(h)$. |
| **Member 4** | Risk Analyst & Backtest Lead | `src/risk_engine.py`<br>`src/backtest.py` | Multiscale VaR/ES engine, benchmark models, Kupiec POF, Christoffersen independence tests, H-TCM formulation. |
| **Member 5** | Visualizer & Report Lead | `src/visualizer.py`<br>`report/report.tex` | 4 high-DPI publication figures, academic report, AI disclosure appendix. |

***

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Imashaidk/QuantEdge1.0.git
cd QuantEdge1.0
pip install -r requirements.txt
```

### 2. Single-Command Full Reproduction
As required by competition guidelines, one command reproduces every number, table, and figure:
```bash
python run_all.py
```
*Runtime: ~15 to 20 seconds.*

### 3. Run Unit Test Suite
```bash
pytest tests/
```
*All 41 tests pass in ~15 seconds.*

### 4. Compiling the LaTeX Report
To compile the academic report into PDF:
- **Local LaTeX:**
  ```bash
  cd report
  pdflatex report.tex
  ```
- **Overleaf:** Upload the repository (or the `report/` and `figures/` directories) to Overleaf and compile with pdfLaTeX.

# QuantEdge 1.0 — Risk Across Tails and Timescales

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Competition](https://img.shields.io/badge/SAIFA-Quant%20Edge%201.0-orange)](https://saifa.lk)

Official repository for **SAIFA Quant Edge 1.0 (Round 1 — Initial Screening Challenge)**.  
**Submission Deadline:** Wednesday, 7 October 2026 - 23:59 (Sri Lanka Time)

---

## 🎯 The Core Research Question
> **"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"**

Standard risk models assume a single horizon and static dependence, scaling 1-day risk via $\sqrt{h}$. In real financial markets, assets co-move differently over days than over months, and joint crash dependence spikes during market turmoil. 

This project constructs **QuantEdge-MTR (Multiscale Tail Risk Framework)**, combining:
1. **Wavelet Multiresolution Analysis (MODWT)** to decompose multi-asset returns into discrete timescales (2 days to $>64$ days).
2. **Semi-Parametric ARMA-GARCH + EVT Margins** to eliminate heteroskedasticity and accurately model extreme heavy tails.
3. **Multiscale Asymmetric Copulas** (Clayton, Gumbel, Student-$t$) to track the evolution of lower tail dependence $\lambda_L(h)$ across investment horizons.
4. **Out-of-Sample Backtesting** against conventional benchmarks (Historical Simulation, Gaussian, and Basel $\sqrt{h}$ scaling) using Kupiec POF and Christoffersen tests.
5. **A Concrete Institutional Policy:** The **Horizon-Conditioned Tail Capital Multiplier (H-TCM)**.

---

## 🏗️ Repository Architecture

```text
QuantEdge/
├── data/                  # Cleaned historical asset price & return CSVs (SPY, QQQ, TLT, GLD, HYG)
├── docs/                  # Competition compliance & AI disclosure log
├── figures/               # Publication-grade figures (300 DPI) for final report
├── report/                # 10-page final submission report (LaTeX / PDF)
├── src/
│   ├── __init__.py
│   ├── config.py          # Portfolio tickers, date windows, hyperparameters, global seed
│   ├── data_loader.py     # Public data ingestion & log return calculation
│   ├── wavelets.py        # MODWT multiresolution decomposition
│   ├── margins.py         # ARMA-GARCH + EVT tail modeling
│   ├── copulas.py         # Copula fitting & tail dependence extraction
│   ├── risk_engine.py     # VaR / ES multiscale calculation
│   ├── backtest.py        # Kupiec, Christoffersen, out-of-sample tests
│   └── visualizer.py      # Publication-grade plotting pipeline
├── tests/                 # Automated unit tests
├── run_all.py             # Single-command runner reproducing all report figures/tables
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 👥 Team Work Breakdown Matrix

| Role | Lead | Target Modules | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Member 1** | Team Lead & Pipeline Architect | `src/config.py`<br>`src/data_loader.py`<br>`run_all.py` | Data fetching/caching, in/out-of-sample splitting, master runner, repo PR review, ZIP packaging. |
| **Member 2** | Wavelet & Signal Specialist | `src/wavelets.py`<br>`tests/test_wavelets.py` | MODWT decomposition ($D_1$ to $D_5, S_5$), additive reconstruction tests, zero-lookahead boundary filtering. |
| **Member 3** | Econometrician & Copula Modeler | `src/margins.py`<br>`src/copulas.py` | ARMA-GARCH + EVT-POT margins, uniform PIT validation, copula MLE fitting, tail dependence curves $\lambda_L(h)$. |
| **Member 4** | Risk Analyst & Backtest Lead | `src/risk_engine.py`<br>`src/backtest.py` | Multiscale VaR/ES engine, benchmark models, Kupiec POF, Christoffersen independence tests, H-TCM formulation. |
| **Member 5** | Visualizer & Report Lead | `src/visualizer.py`<br>`report/report.tex` | 4 high-DPI publication figures, 10-page academic report, AI disclosure appendix. |

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Imashaidk/QuantEdge1.0.git
cd QuantEdge1.0
pip install -r requirements.txt
```

### 2. Single-Command Full Reproduction
As required by the competition guidelines, one command reproduces every number, table, and figure:
```bash
python run_all.py
```

---

## 👥 Contributing & Team Protocol
- Work is organized across dedicated workstream branches (`feat/ws<number>-<topic>`).
- **Never push directly to `main`**; all features must enter via reviewed Pull Requests (PRs).
- Refer to your local `TEAM_EXECUTION_PROMPT.md` for specific role instructions, contracts, and prompt templates.

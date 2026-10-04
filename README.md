# QuantEdge 1.0 — Risk Across Tails and Timescales

**SAIFA Quant Edge 1.0 (Round 1 — Initial Screening Challenge)**  
**Submission Deadline:** Wednesday, 7 October 2026 - 23:59 (Sri Lanka Time)

---

## 🎯 The Core Research Question
> **"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"**

### Background & Objective
Standard market-risk models typically assume:
1. A single investment horizon (e.g., daily returns scaled by $\sqrt{t}$).
2. A static dependence structure (e.g., linear Pearson correlation, or a single copula across all frequencies).

In reality:
- **Tail Dependence**: Assets frequently co-move much more strongly during severe market sell-offs and crashes than during calm periods.
- **Multiscale Dynamics**: Co-movement over high frequencies (intraday to daily noise, liquidity shocks) differs fundamentally from medium frequencies (swing cycles, business quarters) and low frequencies (macroeconomic regime shifts).

This project develops an empirical market-risk framework combining **Wavelet Multiresolution Analysis (MRA / MODWT)** and **Copula Modeling** to quantify how tail dependence evolves across timescales, and evaluates the out-of-sample risk underestimation resulting from ignoring timescale-dependent tail risk.

---

## 🏗️ Repository Architecture (Planned)

```text
QuantEdge/
├── data/                  # Public asset return series & fetch scripts
├── notebooks/             # Exploratory data analysis & visualization
├── src/
│   ├── data_loader.py     # Data retrieval and preprocessing pipeline
│   ├── wavelets.py        # Multiresolution Wavelet Decomposition (MODWT)
│   ├── margins.py         # Marginal distribution fitting (e.g., ARMA-GARCH / EVT)
│   ├── copulas.py         # Copula fitting & tail dependence coefficients
│   ├── risk_engine.py     # Value at Risk (VaR) & Expected Shortfall (ES) calculation
│   └── backtest.py        # Out-of-sample backtesting against simple benchmarks
├── figures/               # Generated figures for the report
├── report/                # 10-page submission report (LaTeX / PDF)
├── run_all.py             # Single command reproduction script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation and reproduction guide
```

---

## ⚖️ Deliverable Checklist (Competition Requirements)
- [ ] **Report**: Max 10 pages PDF (excluding cover & references).
- [ ] **Reproducibility**: One single command reproduces every table and figure.
- [ ] **Benchmark**: Out-of-sample backtest vs a simple benchmark (e.g., standard Historical Simulation / Gaussian / Student-$t$ VaR).
- [ ] **Actionable Takeaway**: One concrete recommendation an institutional risk manager can implement immediately.
- [ ] **Submission Size**: Final ZIP archive $\le$ 25 MB.
- [ ] **AI Disclosure**: Transparent disclosure in appendix / README.

# QuantEdge 1.0: Risk Across Tails and Timescales

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Competition: SAIFA Quant Edge 1.0](https://img.shields.io/badge/Competition-SAIFA%20Quant%20Edge%201.0-orange)](https://saifa.lk)
[![Tests: 45 Passed](https://img.shields.io/badge/tests-45%20passed-success)](tests/)
[![Build: Reproducible](https://img.shields.io/badge/pipeline-10.1s%20execution-brightgreen)](run_all.py)

Official research submission for **SAIFA Quant Edge 1.0: Initial Screening Challenge (Round 1)**.  
**Submission Deadline:** Wednesday, 7 October 2026 at 23:59 (Sri Lanka Time).  
**Repository:** [https://github.com/Imashaidk/QuantEdge1.0.git](https://github.com/Imashaidk/QuantEdge1.0.git)

***

## The Research Question

> *"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"*

Most commercial risk systems scale daily Value-at-Risk (VaR) to multi-day investment horizons using the Basel square-root-of-time formula:

$$\text{VaR}_h = \text{VaR}_1 \times \sqrt{h}$$

This rule assumes that financial returns follow a simple random walk with static correlation and normal distributions. In real financial markets, this assumption breaks down. Assets co-move differently across timescales, and joint crash dependence manifests distinctly across high-frequency rebalancing and macroeconomic holding periods.

This project delivers **QuantEdge-MTR (Multiscale Tail Risk Framework)**, an econometric methodology combining shift-invariant wavelets, extreme value theory, multiscale copulas, and regulatory backtesting to resolve the challenge question empirically.

***

## Core Research Findings

### 1. Persistent Tail Crash Co-dependence Across Timescales
Using rigorous Probability Integral Transform (PIT) uniform margins from AR(1)-GJR-GARCH(1,1) + EVT-POT filtering, empirical lower-tail crash dependence remains persistent across investment horizons: $\hat{\lambda}_L = 0.189$ at daily noise scales ($D_1$, 2 to 4 days) and $\hat{\lambda}_L = 0.201$ at weekly swing scales ($D_2$, 4 to 8 days), gradually settling to $\hat{\lambda}_L = 0.052$ at quarterly horizons ($D_5$, 32 to 64 days). This confirms that cross-asset crash co-dependence does not vanish at multi-day horizons.

### 2. Student-t Copula Dominance Across Horizons
Across all decomposed timescales ($D_1$ through $S_5$), the Student-$t$ copula decisively wins the model tournament evaluated by the Bayesian Information Criterion (BIC), outperforming Gaussian, Clayton, Gumbel, and Frank alternatives. Standardized residuals display symmetric fat tails across frequencies. The Timescale Asymmetry Ratio remains tightly bounded ($\text{TAR} \in [-0.013, +0.060]$), demonstrating that multiscale asset co-dependence is elliptical and fat-tailed.

### 3. Out-of-Sample Backtesting & Basel Scaling Insights
Over 875 out-of-sample trading days (2023 to 2026), conventional square-root-of-time scaling was statistically conservative at 5-day (3 breaches, 0.34%) and 20-day horizons (0 breaches, 0.00% vs ~8.7 expected). However, the 1-day Parametric Gaussian model generates 13 breaches (a 1.49% breach rate, approaching the supervisory penalty boundary), showing that ignoring fat tails understates short-term daily risk.

### 4. An Actionable Solution: The Contingent H-TCM Rule
For risk committees seeking an operational enhancement, we introduce the **Horizon-Conditioned Tail Capital Multiplier (H-TCM)**:

$$\text{VaR}_h^* = \text{VaR}_1 \times \sqrt{h} \times \left[ 1 + \kappa \cdot \max\left(0, \frac{\hat{\lambda}_L(h) - \hat{\lambda}_L(1)}{\hat{\lambda}_L(1) + \epsilon}\right) \right]$$

Rather than imposing an unconditional capital penalty that locks up excessive liquidity during calm markets, H-TCM acts as a contingent policy buffer. When tail dependence at weekly horizons exceeds the daily baseline ($\hat{\lambda}_L(5) = 0.201 > 0.189$), it adds a targeted $+2.2\%$ capital buffer ($\kappa = 0.35$), while reverting to $1.000$ at horizons where tail dependence does not exceed baseline risk.

***

## Empirical Results Summary

### Wavelet Percentage Variance Contribution Across Assets
Decomposed via Maximal Overlap Discrete Wavelet Transform (MODWT, Symlet 8, Level 5):

| Scale | Trading Horizon | SPY | QQQ | TLT | GLD | HYG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 2 to 4 Days (Noise) | 63.95% | 62.86% | 53.38% | 54.42% | 53.62% |
| **D2** | 4 to 8 Days (Weekly) | 17.37% | 19.14% | 27.66% | 21.36% | 18.77% |
| **D3** | 8 to 16 Days (Bi-weekly) | 9.75% | 9.53% | 9.62% | 13.26% | 15.67% |
| **D4** | 16 to 32 Days (Monthly) | 3.92% | 3.83% | 4.28% | 4.72% | 5.61% |
| **D5** | 32 to 64 Days (Quarterly) | 2.45% | 2.22% | 2.02% | 3.06% | 3.56% |
| **S5** | >64 Days (Macro Trend) | 2.56% | 2.41% | 3.03% | 3.18% | 2.77% |

### Copula Tournament Leaderboard Across Horizons
Five copula families fitted via MLE and evaluated by Bayesian Information Criterion (BIC):

| Scale | Trading Horizon | Best Copula | Theo. $\lambda_L$ | Theo. $\lambda_U$ | Emp. $\lambda_L$ | Emp. $\lambda_U$ | Emp. TAR | BIC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 2 to 4 Days | Student-t | 0.020 | 0.020 | 0.189 | 0.187 | +0.002 | -5547.0 |
| **D2** | 4 to 8 Days | Student-t | 0.024 | 0.024 | 0.201 | 0.214 | -0.013 | -4623.9 |
| **D3** | 8 to 16 Days | Student-t | 0.005 | 0.005 | 0.176 | 0.168 | +0.008 | -4545.2 |
| **D4** | 16 to 32 Days | Student-t | 0.000 | 0.000 | 0.081 | 0.057 | +0.025 | -2210.3 |
| **D5** | 32 to 64 Days | Student-t | 0.000 | 0.000 | 0.052 | 0.064 | -0.012 | -1290.4 |
| **S5** | >64 Days | Student-t | 0.046 | 0.046 | 0.178 | 0.118 | +0.060 | -4961.8 |

### Out-of-Sample Backtesting Performance (875 Test Days: 2023 to 2026)
Models calibrated strictly on historical data (2015 to 2022) with zero lookahead bias:

| Horizon | Model | Obs | Breaches | Breach Rate | Kupiec p | Christoffersen p | Basel Zone | FZ Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1d** | Historical Simulation | 875 | 6 | 0.69% | 0.3219 | 0.0286 | GREEN | -3.7191 |
| **1d** | Parametric Gaussian | 875 | 13 | 1.49% | 0.1780 | 0.1824 | GREEN | -3.6786 |
| **1d** | Static Copula | 875 | 5 | 0.57% | 0.1659 | 0.0182 | GREEN | -3.6674 |
| **1d** | Basel Sqrt Time | 875 | 6 | 0.69% | 0.3219 | 0.0286 | GREEN | -3.7191 |
| **1d** | **Proposed Multiscale Copula** | **875** | **5** | **0.57%** | **0.1659** | **0.0182** | **GREEN** | **-3.6674** |
| **1d** | H-TCM Adjusted | 875 | 6 | 0.69% | 0.3219 | 0.0286 | GREEN | -3.7191 |
| **5d** | Historical Simulation | 871 | 2 | 0.23% | 0.0059 | 0.0016 | GREEN | -2.9245 |
| **5d** | Parametric Gaussian | 871 | 6 | 0.69% | 0.3282 | 0.0003 | GREEN | -3.0778 |
| **5d** | Static Copula | 871 | 2 | 0.23% | 0.0059 | 0.0016 | GREEN | -2.8764 |
| **5d** | Basel Sqrt Time | 871 | 3 | 0.34% | 0.0244 | 0.0000 | GREEN | -2.9744 |
| **5d** | **Proposed Multiscale Copula** | **871** | **2** | **0.23%** | **0.0059** | **0.0016** | **GREEN** | **-2.8219** |
| **5d** | H-TCM Adjusted | 871 | 3 | 0.34% | 0.0244 | 0.0000 | GREEN | -2.9261 |
| **20d** | Historical Simulation | 856 | 0 | 0.00% | 0.0000 | 1.0000 | GREEN | -2.3530 |
| **20d** | Parametric Gaussian | 856 | 3 | 0.35% | 0.0274 | 0.0000 | GREEN | -2.6591 |
| **20d** | Static Copula | 856 | 0 | 0.00% | 0.0000 | 1.0000 | GREEN | -2.2601 |
| **20d** | Basel Sqrt Time | 856 | 0 | 0.00% | 0.0000 | 1.0000 | GREEN | -2.3963 |
| **20d** | **Proposed Multiscale Copula** | **856** | **0** | **0.00%** | **0.0000** | **1.0000** | **GREEN** | **-2.2601** |
| **20d** | H-TCM Adjusted | 856 | 0 | 0.00% | 0.0000 | 1.0000 | GREEN | -2.3963 |

### H-TCM Contingent Capital Multiplier Sensitivity Matrix
Values for risk desks across calibration factors $\kappa \in [0.20, 0.50]$ (baseline $\hat{\lambda}_L(1) = 0.189$):

| Horizon | Scale | $\kappa = 0.20$ | $\kappa = 0.35$ (Recommended) | $\kappa = 0.50$ | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **h = 1d** | D1 | 1.000 | **1.000** | 1.000 | Baseline Allocation (0.0% Surcharge) |
| **h = 5d** | D2 | 1.013 | **1.022** | 1.032 | Targeted Contingent Surcharge (+2.2%) |
| **h = 20d** | D4 | 1.000 | **1.000** | 1.000 | No Surcharge Required (0.0% Surcharge) |
| **h = 40d** | D5 | 1.000 | **1.000** | 1.000 | No Surcharge Required (0.0% Surcharge) |

***

## Methodology & Architectural Highlights

1. **Shift-Invariant MODWT Filter Bank:**
   Standard decimated wavelets (DWT) discard half the sample at each scale, leaving only 90 observations at scale 5 from a 2,889-day history. In contrast, the Maximal Overlap Discrete Wavelet Transform (MODWT) preserves all 2,889 daily points across every scale and achieves machine-precision reconstruction:
   $$\max_t \left| R_t - \left( \sum_{j=1}^5 D_{j,t} + S_{5,t} \right) \right| = 3.77 \times 10^{-14} \ll 10^{-10}$$

2. **Two-Stage Semi-Parametric Margins:**
   Raw returns cannot be plugged directly into copulas due to volatility clustering. Each wavelet series is filtered with an AR(1)-GJR-GARCH(1,1) model with Student-t innovations to capture leverage asymmetry. The standardized filtered residuals $z_t = \epsilon_t / \sigma_t$ are then modeled using Extreme Value Theory (EVT) Peaks-Over-Threshold: an empirical distribution on the central 80% and Generalized Pareto Distributions (GPD) on the extreme 10% tails, generating strict $\text{Uniform}(0, 1)$ margins.

3. **Scale-Optimal Copula Tournament:**
   Fits five copula families (Gaussian, Student-t, Clayton, Gumbel, Frank) via Maximum Likelihood Estimation at each scale and selects the best model using Bayesian Information Criterion (BIC).

4. **Rigorous Regulatory Backtest Suite:**
   Evaluates unconditional coverage (Kupiec POF LR test), conditional coverage (Christoffersen independence test), official Basel Committee Traffic Light zone classification, and joint elicitable scoring via Fissler-Ziegel (FZ) loss at unified $\alpha = 0.99$.

***

## Project Directory Layout

```text
QuantEdge/
├── data/
│   ├── raw_prices.csv           # Historical daily adjusted close prices (2015 to 2026)
│   └── log_returns.csv          # Log return matrix (2,889 observations x 5 assets)
├── docs/
│   └── AI_DISCLOSURE.md         # Competition compliance & AI disclosure log
├── figures/
│   ├── fig1_wavelet_mra_decomposition.png     # 300 DPI: MODWT multiresolution series
│   ├── fig2_tail_dependence_vs_horizon.png    # 300 DPI: lambda_L, lambda_U, and TAR curve
│   ├── fig3_backtest_var_exceedances.png      # 300 DPI: 875-day out-of-sample loss breaches
│   └── fig4_regulatory_traffic_light.png      # 300 DPI: Basel Traffic Light performance
├── report/
│   ├── report.tex               # Formal academic manuscript (LaTeX source)
│   └── REPORT.md                # Markdown companion report
├── src/
│   ├── __init__.py
│   ├── config.py                # Universe tickers, weights, dates, hyperparameters
│   ├── data_loader.py           # Ingestion, log returns, ADF stationarity, caching
│   ├── wavelets.py              # MODWT filter bank, additive MRA, variance decomposition
│   ├── margins.py               # AR(1)-GJR-GARCH(1,1) + EVT-POT GPD tails + PIT validation
│   ├── copulas.py               # 5-family tournament, MLE fitting, BIC selection
│   ├── risk_engine.py           # Multiscale VaR/ES, benchmark models, H-TCM rule
│   ├── backtest.py              # Kupiec, Christoffersen, Basel zones, FZ scoring
│   └── visualizer.py            # High-resolution plotting and LaTeX table exporter
├── tables/
│   ├── backtest_metrics.tex     # Auto-generated backtesting summary table
│   ├── copula_tournament.tex    # Auto-generated copula tournament table
│   └── variance_decomposition.tex # Auto-generated multiresolution variance table
├── tests/
│   ├── test_margins_copulas.py  # Unit tests for GARCH, EVT, and copulas
│   ├── test_risk_backtest.py    # Unit tests for risk metrics and backtests
│   ├── test_visualizer.py       # Unit tests for plotting and tables
│   └── test_wavelets.py         # Unit tests for MODWT and additive reconstruction
├── run_all.py                   # Master single-command reproduction script
├── verify_submission.py         # Automated 7-criteria pre-submission verification
├── requirements.txt             # Dependency definitions
└── README.md                    # Project documentation
```

***

## Team Work Breakdown Matrix

| Role | Lead | Target Modules | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Member 1** | Team Lead & Pipeline Architect | `src/config.py`<br>`src/data_loader.py`<br>`run_all.py` | Data fetching and caching, train/test splitting, master runner, verification suite, ZIP packaging. |
| **Member 2** | Wavelet & Signal Specialist | `src/wavelets.py`<br>`tests/test_wavelets.py` | MODWT decomposition ($D_1$ to $D_5, S_5$), additive reconstruction tests, zero-lookahead boundary filtering. |
| **Member 3** | Econometrician & Copula Modeler | `src/margins.py`<br>`src/copulas.py` | AR(1)-GJR-GARCH(1,1) + EVT-POT margins, uniform PIT validation, copula MLE fitting, tail dependence curves. |
| **Member 4** | Risk Analyst & Backtest Lead | `src/risk_engine.py`<br>`src/backtest.py` | Multiscale VaR/ES engine, benchmark models, Kupiec POF, Christoffersen independence tests, H-TCM formulation. |
| **Member 5** | Visualizer & Report Lead | `src/visualizer.py`<br>`report/report.tex` | High-DPI publication figures, academic manuscript, AI disclosure appendix. |

***

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Imashaidk/QuantEdge1.0.git
cd QuantEdge1.0
pip install -r requirements.txt
```

### 2. Single-Command Full Reproduction
As required by competition guidelines, one command executes the entire pipeline and regenerates all figures, tables, and metrics:
```bash
python run_all.py
```
*Total execution time: ~10 seconds (< 3 minutes hard limit).*

### 3. Automated Test Suite
Run the test suite to verify all mathematical and econometric invariants:
```bash
pytest tests/
```
*All 45 tests pass in ~15 seconds.*

### 4. Automated Submission Verification
```bash
python verify_submission.py
```
*Verifies all 7 audit integrity criteria (data splits, observation counts, theoretical tail bounds, H-TCM multipliers, confidence levels, LaTeX tables, zip size, zero emojis/non-ASCII characters).*

### 5. Compiling the LaTeX Report
To generate the final academic PDF manuscript:
- **Local compilation:**
  ```bash
  cd report
  pdflatex report.tex
  ```
- **Overleaf:** Upload the repository folder to Overleaf and compile using standard pdfLaTeX.

***

## References

1. Basel Committee on Banking Supervision. (1996). *Supervisory Framework for the Use of "Backtesting" in Conjunction with the Internal Models Approach to Market Risk Capital Requirements*. Basel: Bank for International Settlements. https://www.bis.org/publ/bcbs22.htm
2. Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841-862. https://doi.org/10.2307/2527341
3. Embrechts, P., McNeil, A., & Straumann, D. (2002). Correlation and dependence in risk management: Properties and pitfalls. In M. A. H. Dempster (Ed.), *Risk Management: Value at Risk and Beyond* (pp. 176-223). Cambridge: Cambridge University Press. https://doi.org/10.1017/CBO9780511615337.008
4. Fissler, T., & Ziegel, J. F. (2016). Higher order elicitability and prediction markets of higher order. *The Annals of Statistics*, 44(5), 2153-2181. https://doi.org/10.1214/16-AOS1439
5. Gencay, R., Selcuk, F., & Whitcher, B. (2001). *An Introduction to Wavelets and Other Filtering Methods in Finance and Economics*. San Diego: Academic Press.
6. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779-1801. https://doi.org/10.1111/j.1540-6261.1993.tb05128.x
7. Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73-84. https://doi.org/10.3905/jod.1995.407942
8. McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools* (Revised ed.). Princeton, NJ: Princeton University Press.
9. Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time Series Analysis*. Cambridge: Cambridge University Press. https://doi.org/10.1017/CBO9780511841040
10. Pickands, J. (1975). Statistical inference using extreme order statistics. *The Annals of Statistics*, 3(1), 119-131. https://doi.org/10.1214/aos/1176343003
11. Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Universite de Paris*, 8, 229-231.

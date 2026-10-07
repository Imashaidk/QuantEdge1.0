# Risk Across Tails and Timescales: Frequency-Variant Tail Dependence and Limitations of Horizon Square-Root Scaling

> **SAIFA Quant Edge 1.0: Round 1 Submission**  
> **Official Challenge Question:**  
> *"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"*  
> **Repository:** [https://github.com/Imashaidk/QuantEdge1.0.git](https://github.com/Imashaidk/QuantEdge1.0.git)

***

## Executive Summary

Most banks and asset managers scale 1-day Value-at-Risk ($\text{VaR}_1$) numbers to multi-day horizons using the classic Basel square-root-of-time rule:
$$\text{VaR}_h = \text{VaR}_1 \times \sqrt{h}$$
This rule assumes that financial returns are i.i.d. normal and that joint dependence between assets is static across holding periods, whether assets are held for two hours or three months.

In this study, we answer the competition question directly: **tail dependence is persistent and horizon-dependent**. Assets that appear weakly correlated in daily noise series maintain non-trivial crash co-dependence across multi-day and multi-week investment horizons.

### Key Discoveries:
1. **Multiscale Tail Dependence & The Gaussian Benchmark:** Empirical lower-tail crash dependence varies across timescales: highest at short frequencies ($D_1 \approx 0.189, D_2 \approx 0.201, D_3 \approx 0.176$), and dropping noticeably at longer monthly and macroeconomic scales ($D_4 \approx 0.098, D_5 \approx 0.060, S_5 \approx 0.099$). Crucially, benchmarking against a Gaussian copula with matched correlation reveals that high-frequency co-exceedance ($D_1$, $\hat{\lambda}_L = 0.189$ vs Gaussian benchmark $0.185$) is largely accounted for by linear correlation (excess $+0.004$), with positive excess observed at $D_1$ to $D_3$ (peaking at $D_2$, excess $+0.030$, though overlapping bootstrap intervals show this peak is within estimation noise), while excess turns slightly negative at longer scales ($D_4, D_5, S_5$, between $-0.011$ and $-0.022$).
2. **Student-t Copula Dominance:** The Student-$t$ copula decisively wins the model tournament across all timescales by Bayesian Information Criterion (BIC), indicating that multi-asset joint tail risk is characterized by symmetric, elliptical fat-tailed extreme co-movements rather than Archimedean asymmetry.
3. **The Timescale Asymmetry Ratio ($\text{TAR} = \hat{\lambda}_L - \hat{\lambda}_U$):** Across all decomposed scales, empirical tail asymmetry remains tightly bounded around zero ($\text{TAR} \in [-0.013, +0.011]$), confirming that cross-asset extreme co-movement is elliptical and fat-tailed rather than single-parameter asymmetric.
4. **Out-of-Sample Backtesting & Basel Scaling Insights:** In out-of-sample backtesting across 875 trading days (2023 to 2026), conventional square-root-of-time scaling was statistically conservative at 5-day (3 breaches, 0.34%) and 20-day horizons (0 breaches, 0.00% vs ~8.7 expected). However, the Parametric Gaussian model generates 13 breaches at the 1-day horizon (a 1.49% breach rate vs 1.00% expected). While failing to reject at 5% significance under Kupiec POF ($p = 0.178$), it generates 2.6x the breaches of copula models (5 breaches), showing that assuming normality understates short-term tail risk.
5. **Actionable Fix for Risk Desks (The Contingent H-TCM Rule):** We introduce the **Horizon-Conditioned Tail Capital Multiplier (H-TCM)**:
   $$\text{VaR}_h^* = \text{VaR}_1 \times \sqrt{h} \times \left[ 1 + \kappa \cdot \max\left(0, \frac{\hat{\lambda}_L(h) - \hat{\lambda}_L(1)}{\hat{\lambda}_L(1) + \epsilon}\right) \right]$$
   H-TCM operates as a contingent macroprudential buffer. When tail dependence at weekly horizons exceeds the daily baseline ($\hat{\lambda}_L(5) = 0.201 > 0.189$), it adds a targeted $+2.2\%$ precautionary capital buffer ($\kappa = 0.35$), while reverting to $1.000$ at horizons where tail dependence does not exceed baseline levels.

***

## 1. Asset Universe and Economic Intuition

We examine a five-pillar institutional portfolio across 2,889 trading days (January 2, 2015 to June 30, 2026):
- **SPY (30%):** S&P 500 ETF (Core equity beta).
- **QQQ (20%):** Invesco QQQ Trust (High-beta growth and technology).
- **TLT (25%):** 20+ Year Treasury Bond ETF (Duration hedge and flight to safety).
- **GLD (15%):** SPDR Gold Shares (Inflation hedge and store of value).
- **HYG (10%):** High Yield Corporate Bond ETF (Credit spread and liquidity risk).

Testing a balanced multi-asset portfolio matters because pairs of equities frequently move together. Only a multi-asset universe containing equities, duration, gold, and credit reveals how flight-to-safety hedging behaviors evolve across short-term noise and macroeconomic cycles.

***

## 2. Methodology: MODWT, EVT Margins, and Copula Tournament

```text
QuantEdge-MTR Pipeline Architecture
===================================================================================
 [Data Layer]       Multi-Asset Log Returns (SPY, QQQ, TLT, GLD, HYG: 2015 to 2026)
                           │
                           ▼
 [Wavelet Layer]    MODWT Additive MRA (Scales D1: 2-4d up to D5: 32-64d + S5)
                    Additive Invariant: max |R - sum(D_j) - S_5| < 1e-13
                           │
                           ▼
 [Margin Layer]     AR(1)-GJR-GARCH(1,1) + EVT-POT (GPD Tails) + PIT
                    Uniform PIT Validation: u ~ Uniform(0, 1)
                           │
                           ▼
 [Copula Layer]     Scale-Optimal Tournament (Gaussian, t, Clayton, Gumbel, Frank)
                           │
                           ▼
 [Risk Engine]      VaR (95%, 99%), ES (99%), 5 Models, and H-TCM Rule
                           │
                           ▼
 [Backtest Layer]   Kupiec POF, Christoffersen Independence, Basel Traffic Light
===================================================================================
```

### 2.1 Why We Use MODWT Instead of Standard DWT
Standard Discrete Wavelet Transform (DWT) downsamples the series by half at each decomposition scale. By scale 5, 31 out of every 32 observations are discarded, leaving less than 90 points from a 2,889-day history. Estimating extreme tail copulas on 90 points is statistically fragile.

In contrast, the **Maximal Overlap Discrete Wavelet Transform (MODWT)** is shift-invariant and preserves the full sample of 2,889 daily points across every scale. It reconstructs original price returns with machine precision:
$$\max_t \left| R_t - \left( \sum_{j=1}^5 D_{j,t} + S_{5,t} \right) \right| = 3.77 \times 10^{-14} \ll 10^{-10}$$

### 2.2 Volatility Filtering via GARCH and EVT Margins
Calculating copula dependence directly on raw returns creates spurious correlation caused by volatility clustering. We filter each asset detail series with an $\text{AR}(1)\text{-GJR-GARCH}(1,1)$ model with Student-$t$ innovations to remove serial correlation and volatility clustering while accommodating the leverage effect.

We then apply Extreme Value Theory (EVT) Peaks-Over-Threshold to the standardized filtered residuals $z_t = \epsilon_t / \sigma_t$. Generalized Pareto Distributions (GPD) are fitted to the lower 10% and upper 10% tails, while the empirical CDF models the interior 80%. Transforming through the estimated CDF produces strictly valid $\text{Uniform}(0, 1)$ margins verified by probability integral transform diagnostics.

***

## 3. Empirical Results

### 3.1 Percentage Variance Contribution by Timescale
Decomposed via MODWT (Symlet 8, Level 5):

| Scale | Period | SPY | QQQ | TLT | GLD | HYG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 2 to 4 days (Noise) | 63.95% | 62.86% | 53.38% | 54.42% | 53.62% |
| **D2** | 4 to 8 days (Weekly) | 17.37% | 19.14% | 27.66% | 21.36% | 18.77% |
| **D3** | 8 to 16 days (Bi-weekly) | 9.75% | 9.53% | 9.62% | 13.26% | 15.67% |
| **D4** | 16 to 32 days (Monthly) | 3.92% | 3.83% | 4.28% | 4.72% | 5.61% |
| **D5** | 32 to 64 days (Quarterly) | 2.45% | 2.22% | 2.02% | 3.06% | 3.56% |
| **S5** | >64 days (Macro trend) | 2.56% | 2.41% | 3.03% | 3.18% | 2.77% |

High-frequency noise ($D_1$) accounts for 53% to 64% of total return variance, while quarterly and macroeconomic cycles ($D_5$ and $S_5$) account for 5% to 6%.

### 3.2 Copula Tournament Leaderboard Across Horizons
Five copula families fitted via MLE and evaluated by Bayesian Information Criterion (BIC), benchmarked against a bivariate Gaussian copula with matched linear correlation:

| Scale | Trading Horizon | Best Copula | Theo. $\lambda_L$ | Theo. $\lambda_U$ | Emp. $\lambda_L$ | 95% CI | Gauss Bench | Excess $\lambda_L$ | Emp. $\lambda_U$ | Emp. TAR | BIC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 2 to 4 Days | Student-t | 0.089 | 0.089 | 0.189 | [0.143, 0.239] | 0.185 | +0.004 | 0.187 | +0.002 | -5547.0 |
| **D2** | 4 to 8 Days | Student-t | 0.079 | 0.079 | 0.201 | [0.154, 0.233] | 0.171 | +0.030 | 0.214 | -0.013 | -4623.4 |
| **D3** | 8 to 16 Days | Student-t | 0.047 | 0.047 | 0.176 | [0.132, 0.217] | 0.166 | +0.009 | 0.168 | +0.008 | -4543.0 |
| **D4** | 16 to 32 Days | Student-t | 0.020 | 0.020 | 0.098 | [0.062, 0.143] | 0.109 | -0.011 | 0.096 | +0.002 | -3043.0 |
| **D5** | 32 to 64 Days | Student-t | 0.001 | 0.001 | 0.060 | [0.019, 0.077] | 0.074 | -0.014 | 0.049 | +0.011 | -1446.6 |
| **S5** | >64 Days | Student-t | 0.055 | 0.055 | 0.099 | [0.026, 0.162] | 0.122 | -0.022 | 0.098 | +0.001 | -4300.5 |

**Disentangling Tail Dependence from Linear Correlation:**
At a $5\%$ quantile threshold ($q = 0.05$), assets governed by a Gaussian copula produce positive co-exceedance due to linear correlation $\rho$. By benchmarking against the bivariate normal integral $\lambda_{\text{Gauss}}(0.05; \rho) = \Phi_2(z_{0.05}, z_{0.05}; \rho) / 0.05$, we find:
- At $D_1$, empirical $\hat{\lambda}_L = 0.189$ vs Gaussian benchmark $0.185$ (excess $+0.004$, 95% CI $[0.143, 0.239]$), consistent with daily co-exceedance being driven primarily by linear correlation rather than non-linear crash dependence.
- At $D_2$, empirical $\hat{\lambda}_L = 0.201$ vs Gaussian benchmark $0.171$ (excess $+0.030$, 95% CI $[0.154, 0.233]$). Moving-block bootstrap analysis (block size 64 days, 300 replications) yields intervals overlapping with D1 ($[0.143, 0.239]$), indicating that the 0.012 gap between D1 and D2 lies within finite-sample estimation uncertainty.
- At $D_4$ and $D_5$, co-exceedance attenuates substantially to $0.098$ and $0.060$, falling below the Gaussian baseline (excess $-0.011$ and $-0.014$).
- At $S_5$, co-exceedance is $0.099$ ($95\%$ CI $[0.026, 0.162]$) with Gaussian benchmark $0.122$ (excess $-0.022$). The empirical data do not support a secular rebound; tail co-dependence remains lower at long holding periods.

***

## 4. Out-of-Sample Backtesting (2023 to 2026)

We evaluate models across out-of-sample trading days (January 2023 to June 2026) for horizons $h \in \{1, 5, 20\}$ days, calibrated strictly on in-sample data from 2015 to 2022:

| Horizon | Model | Obs | Breaches | Breach Rate | Kupiec p | Christoffersen p | FZ Loss | Status / Diagnostic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1d** | Historical Simulation | 875 | 6 | 0.69% | 0.3219 | 0.0286 | -3.7191 | GREEN (Basel) |
| **1d** | Parametric Gaussian | 875 | 13 | 1.49% | 0.1780 | 0.1824 | -3.6786 | GREEN (Basel) |
| **1d** | Static Copula | 875 | 5 | 0.57% | 0.1659 | 0.0182 | -3.6674 | GREEN (Basel) |
| **1d** | Basel Sqrt Time | 875 | 6 | 0.69% | 0.3219 | 0.0286 | -3.7191 | GREEN (Basel) |
| **1d** | Proposed Multiscale Copula | 875 | 5 | 0.57% | 0.1659 | 0.0182 | -3.6674 | GREEN (Basel) |
| **1d** | H-TCM Adjusted | 875 | 6 | 0.69% | 0.3219 | 0.0286 | -3.7191 | GREEN (Basel) |
| **5d** | Historical Simulation | 871 | 2 | 0.23% | 0.0059 | 0.0016 | -2.9245 | GREEN (Diag) |
| **5d** | Parametric Gaussian | 871 | 6 | 0.69% | 0.3282 | 0.0003 | -3.0778 | GREEN (Diag) |
| **5d** | Static Copula | 871 | 2 | 0.23% | 0.0059 | 0.0016 | -2.8764 | GREEN (Diag) |
| **5d** | Basel Sqrt Time | 871 | 3 | 0.34% | 0.0244 | <0.0001 | -2.9744 | GREEN (Diag) |
| **5d** | Proposed Multiscale Copula | 871 | 2 | 0.23% | 0.0059 | 0.0016 | -2.8600 | GREEN (Diag) |
| **5d** | H-TCM Adjusted | 871 | 3 | 0.34% | 0.0244 | <0.0001 | -2.9599 | GREEN (Diag) |
| **20d** | Historical Simulation | 856 | 0 | 0.00% | <0.0001 | 1.0000 | -2.3530 | GREEN (Diag) |
| **20d** | Parametric Gaussian | 856 | 3 | 0.35% | 0.0274 | <0.0001 | -2.6591 | GREEN (Diag) |
| **20d** | Static Copula | 856 | 0 | 0.00% | <0.0001 | 1.0000 | -2.2601 | GREEN (Diag) |
| **20d** | Basel Sqrt Time | 856 | 0 | 0.00% | <0.0001 | 1.0000 | -2.3963 | GREEN (Diag) |
| **20d** | Proposed Multiscale Copula | 856 | 0 | 0.00% | <0.0001 | 1.0000 | -2.2601 | GREEN (Diag) |
| **20d** | H-TCM Adjusted | 856 | 0 | 0.00% | <0.0001 | 1.0000 | -2.3963 | GREEN (Diag) |

> **Interpretation note (FZ Loss: lower is better):** At 1d and 20d, Proposed Multiscale Copula is identical to Static Copula (same breach count and FZ loss). At 5d, Proposed ties on breaches (2) but its FZ loss (-2.8600) is higher (worse) than Historical Simulation (-2.9245), Basel (-2.9744), and Gaussian (-3.0778). The Proposed model is **comparable** to benchmarks, not superior, during this calm 2023-2026 window. The H-TCM buffer is a precautionary overlay for tail-stress regimes, not a proven performance improvement over the observed period. **Numbers above reflect the reference environment -- run `python run_all.py` for exact values in your local environment.**

**Backtest Insights & Model Comparison:**
- At 1-day, the Parametric Gaussian model produces 13 breaches (1.49% failure rate vs 1.00% expected). While failing to reject at 5% significance under Kupiec POF ($p = 0.178$), it generates 2.6x the breach count of copula models (5 breaches), demonstrating substantial tail vulnerability. The Proposed Multiscale Copula is identical to Static Copula at 1-day (5 breaches, FZ loss -3.6674).
- At 5-day, the Proposed Multiscale Copula ties with Historical Simulation and Static Copula for lowest breach count (2 breaches each, both Diagnostic Green). Basel square-root scaling achieves the lowest Fissler-Ziegel score (-2.9744) followed by H-TCM (-2.9599) and Proposed Copula (-2.8600).
- At 20-day, the Proposed Multiscale Copula matches Static Copula (0 breaches, FZ loss -2.2601) and ties with Historical Simulation and Basel scaling at 0 breaches.
- Multi-day square-root scaling was statistically conservative (0 to 3 breaches vs ~8.7 expected), rather than understating risk, due to the steady upward equity trend and overlapping rolling returns.

***

## 5. Practical Policy for Risk Desks: The H-TCM Rule

### The Closed-Form Formula:
$$\text{VaR}_h^* = \text{VaR}_1 \times \sqrt{h} \times \left[ 1 + \kappa \cdot \max\left(0, \frac{\hat{\lambda}_L(h) - \hat{\lambda}_L(1)}{\hat{\lambda}_L(1) + \epsilon}\right) \right]$$

### Sensitivity Table Across Horizons ($\kappa \in [0.20, 0.50]$):
| Horizon | Scale | $\kappa = 0.20$ | $\kappa = 0.35$ (Recommended) | $\kappa = 0.50$ | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$h = 1$d** | D1 | 1.000 | **1.000** | 1.000 | Basel-style Green Zone (Baseline Allocation) |
| **$h = 5$d** | D2 | 1.013 | **1.022** | 1.032 | Precautionary Buffer (+2.2% Contingent Overlay) |
| **$h = 20$d** | D4 | 1.000 | **1.000** | 1.000 | Baseline Scaling (No Surcharge Required) |
| **$h = 40$d** | D5 | 1.000 | **1.000** | 1.000 | Baseline Scaling (No Surcharge Required) |

**Candid Evaluation & Empirical Scope:**
1. **Precautionary Buffer vs Noise:** The +2.2% buffer reflects an empirical difference of $\hat{\lambda}_L(D_2) - \hat{\lambda}_L(D_1) = 0.201 - 0.189 = 0.012$, which operates as a precautionary safety margin rather than a statistically separated divergence.
2. **Calm Market Performance:** In the 2023-2026 backtest, standard scaling was already conservative (3 breaches at 5d). H-TCM held extra capital without reducing breaches below 3, incurring a slightly higher Fissler-Ziegel loss (-2.9599 vs -2.9744).
3. **Long Horizon Behavior:** At 20d and 40d, the multiplier floors at 1.000, identical to standard scaling.
4. **Crisis Positioning:** H-TCM is a contingent stress buffer for systemic crises (such as 2008 or March 2020), and remains untested out-of-sample in a severe historical liquidity shock.

***

## 6. Reproduction, Verification & AI Disclosure

1. **Master Pipeline Reproduction:**
   ```bash
   python run_all.py
   ```
   Execution completes in ~10 seconds (< 3 minutes hard limit) and regenerates all figures and LaTeX tables.
2. **Automated Unit Tests:**
   ```bash
   pytest tests/
   ```
   All 48 unit/integration tests pass in ~10 seconds.
3. **Automated Verification:**
   ```bash
   python verify_submission.py
   ```
   Verifies all submission integrity criteria.
4. **Transparent AI Disclosure:**
   In compliance with competition guidelines, generative AI tools were used materially for technical literature synthesis, Python code drafting/refactoring across `src/`, numerical optimization debugging, iterative repository audits and fixes, and LaTeX table formatting. The human team directed project scoping, audited all code, conducted econometric reality checks (including formulating the Gaussian benchmark and candid H-TCM evaluation), and accepts full intellectual responsibility (see `docs/AI_DISCLOSURE.md`).

***

## 7. References

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

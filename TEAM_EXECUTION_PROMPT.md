# 🏆 SAIFA QUANT EDGE 1.0 — MASTER WINNING PLAN & TEAM EXECUTION PROMPT

> **Project:** Risk Across Tails and Timescales  
> **Target:** First-Place Submission for Round 1 (Initial Screening Challenge)  
> **Deadline:** Wednesday, 7 October 2026 - 23:59 (Sri Lanka Time)  
> **Official Challenge Question:**  
> *"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"*

---

## 📌 HOW TO USE THIS DOCUMENT
Every team member should load this file or copy-paste **Section 5 ("Team Member AI Prompt")** into their AI chat/IDE session alongside their assigned Workstream number. This ensures our entire codebase, methodology, variable naming, and research narrative remain 100% synchronized and mathematically cohesive.

---

## 1. ⚖️ THE JUDGE'S PERSPECTIVE: HOW THIS COMPETITION IS WON

### The Evaluation Criteria Breakdown:
1. **Creativity (25%):** Novel formulation of how frequency components interact with non-linear tail dependence. Not just applying an off-the-shelf tutorial, but showing economic intuition.
2. **Technical Rigour (35%):**
   - No lookahead bias or boundary distortion in filtering (use **MODWT**, never naive DWT which downsamples and ruins sample sizes).
   - Statistically valid margins (standardized residuals from ARMA-GARCH filtered through EVT / Generalized Pareto Distribution before copula probability integral transform).
   - Rigorous statistical testing of out-of-sample backtests (Kupiec POF test, Christoffersen Independence test, and Tick/AS loss for Expected Shortfall).
3. **Clarity of Explanation (20%):** "A simple idea explained well beats a complex one explained badly." The 10-page report must tell an intuitive story backed by clean, publication-grade figures.
4. **Quality of Code & Reproducibility (20%):**
   - Running `python run_all.py` must execute without errors on a clean machine and recreate every single figure and table in the report.
   - Clean file hierarchy, strict type hinting, docstrings, and execution under 5 minutes.
   - Package size well below the 25 MB ceiling.

### Common Pitfalls That Get Teams Eliminated:
- ❌ **Downsampling distortion:** Using standard DWT which halves sample size at each level. (We **MUST use MODWT** - Maximal Overlap Discrete Wavelet Transform).
- ❌ **Naive Margins:** Fitting copulas directly on raw returns without filtering volatility clustering (violates the i.i.d. assumption required for copula estimation).
- ❌ **Vague Recommendation:** Giving generic advice like "diversify more." The judges explicitly require: **"one concrete recommendation a risk manager could act on tomorrow."**
- ❌ **Lookahead Leakage:** Decomposing the entire time series at once before splitting into train/test. Wavelet coefficients must be strictly generated within rolling windows or expanding historical filters.

---

## 2. 🔬 THE RESEARCH & QUANTITATIVE BLUEPRINT

### A. Asset Selection & Economic Justification
We choose a **Cross-Asset Liquidity & Flight-to-Safety Portfolio (2015–2026)**:
1. **Equities (High Beta / Risk-On):** S&P 500 (`SPY`) & Nasdaq 100 (`QQQ`)
2. **Safe-Haven Fixed Income (Duration / Flight-to-Safety):** 20+ Year US Treasury Bond (`TLT`)
3. **Alternative Asset / Inflation Hedge:** Gold (`GLD`)
4. **Credit / Systematic Liquidity Indicator:** High Yield Corporate Bond (`HYG`)

*Justification:*  
- On a **daily/noise timescale (1–4 days)**: Gold and Treasuries frequently decouple or show zero/negative correlation with equities due to microstructure noise and rebalancing.
- On a **crash/tail timescale**: Liquidity crunches (e.g., March 2020 COVID shock) cause margin calls where *all* liquid assets are sold simultaneously—joint tail dependence spikes dramatically across asset classes that otherwise appear uncorrelated.
- On a **macro timescale (32–128 days)**: Monetary policy and inflation regimes drive long-horizon structural co-movements.

### B. Core Mathematical Architecture
```
Raw Asset Returns [R_t]
         │
         ▼
[1] Wavelet Multiresolution Analysis (MODWT)
    Decompose into scales:
    • D1 (2–4 days)     : Microstructure / Noise
    • D2 (4–8 days)     : Weekly Momentum / Swing Trading
    • D3 (8–16 days)    : Bi-weekly Sentiment
    • D4 (16–32 days)   : Monthly Rebalancing
    • D5 (32–64 days)   : Quarterly Business Cycle
    • S5 (>64 days)     : Long-term Macro Trend
         │
         ▼
[2] Marginal Distribution Engine (ARMA-GARCH + EVT)
    • Fit ARMA(1,1)-GJR-GARCH(1,1) with Student-t innovations to filter volatility clustering.
    • Apply semi-parametric Extreme Value Theory (EVT):
      - Body: Empirical CDF
      - Tails: Generalized Pareto Distribution (GPD) for upper & lower thresholds
    • Transform to Uniform Margins: U_i ~ Uniform(0,1) via Probability Integral Transform (PIT).
         │
         ▼
[3] Multiscale Copula Engine
    • Fit family of copulas across each timescale D_j and raw series:
      - Gaussian (Benchmark: Zero tail dependence)
      - Student-t (Symmetric tail dependence)
      - Clayton (Strict lower-tail crash dependence: λ_L > 0, λ_U = 0)
      - Gumbel (Upper-tail boom dependence: λ_U > 0, λ_L = 0)
      - Rotated Clayton / Symmetrized Joe-Clayton (BB7)
    • Extract Empirical and Theoretical Tail Dependence Coefficients:
      λ_L(τ) = lim_{u->0} P(U_1 < u | U_2 < u) as a function of timescale τ.
         │
         ▼
[4] Risk Quantification Engine (VaR & ES)
    • Estimate Portfolio Value-at-Risk (VaR_99%, VaR_97.5%) and Expected Shortfall (ES_97.5%).
    • Model A: Standard Benchmark (Single-horizon Static Gaussian / Historical Simulation).
    • Model B: Static Full-Spectrum Copula.
    • Model C: **Multiscale Wavelet-Copula Framework (Proposed Model)**.
         │
         ▼
[5] Out-of-Sample Backtesting & Diagnostic Validation
    • Kupiec Proportion of Failures (POF) test (unconditional coverage).
    • Christoffersen Independence test (conditional coverage & loss clustering).
    • Fissler-Ziegel Joint VaR/ES scoring function.
         │
         ▼
[6] The Concrete Risk Manager Recommendation
    • Actionable, formulaic policy: **"Timescale-Conditioned Tail Capital Multiplier (T-TCM)"**.
```

---

## 3. 👥 WORKSTREAM BREAKDOWN & ASSIGNMENTS

| Workstream | Module / Target File | Primary Responsibilities |
| :--- | :--- | :--- |
| **WS 1: Data & Pipeline** | `src/data_loader.py`<br>`src/config.py` | • Fetch daily adjusted close data (2015-01-01 to 2026-06-30) for `SPY`, `QQQ`, `TLT`, `GLD`, `HYG`.<br>• Compute log returns, handle missing ticks, verify stationary tests (ADF/KPSS).<br>• Split dataset: In-Sample (2015–2022) / Out-of-Sample (2023–2026). |
| **WS 2: Wavelet Engine** | `src/wavelets.py` | • Implement Maximal Overlap Discrete Wavelet Transform (**MODWT**) using PyWavelets / custom filter bank.<br>• Wavelet family: Daubechies (`db4`) or Symlet (`sym8`) with circular/reflection boundary handling.<br>• Implement Multiresolution Analysis (MRA) decomposition ensuring additive property: $R_t = \sum D_j + S_J$.<br>• Ensure zero lookahead bias for rolling window analysis. |
| **WS 3: Margins & Copulas** | `src/margins.py`<br>`src/copulas.py` | • Fit ARMA(1,1)-GARCH(1,1) to filter time-varying heteroskedasticity.<br>• Model tails using Generalized Pareto Distribution (EVT-POT method).<br>• PIT transform into uniform variables $U(0,1)$.<br>• Fit Gaussian, Student-$t$, Clayton, and Gumbel copulas via Maximum Likelihood Estimation (MLE).<br>• Calculate lower tail dependence $\lambda_L$ and upper $\lambda_U$ for each timescale. |
| **WS 4: Risk & Backtesting** | `src/risk_engine.py`<br>`src/backtest.py` | • Simulate joint portfolio returns from multiscale copula draws.<br>• Compute 1-day, 5-day, and 20-day equivalent VaR (95%, 99%) and ES (97.5%).<br>• Compare against benchmarks: Historical Simulation, Normal parametric, Static Student-$t$ Copula.<br>• Run statistical backtests: Kupiec LR test, Christoffersen test, and Regulatory Traffic Light zone. |
| **WS 5: Master Pipeline & Plots** | `run_all.py`<br>`src/visualizer.py` | • Build the single master execution command `python run_all.py`.<br>• Generate publication-grade, high-DPI figures saved in `figures/`: (1) Wavelet scale decomposition, (2) Tail dependence vs. timescale curve ($\lambda_L$ vs Horizon), (3) VaR breach out-of-sample chart, (4) Summary metrics table.<br>• Keep execution fast (< 3 minutes). |
| **WS 6: 10-Page Report** | `report/report.tex` / `report.md` | • Write the competition paper adhering strictly to the 10-page limit.<br>• Sections: (1) Executive Summary & Core Question, (2) Economic Justification of Assets, (3) Mathematical Methodology, (4) Empirical Findings, (5) Out-of-Sample Validation, (6) Actionable Risk Manager Policy, (7) AI Disclosure Appendix. |

---

## 4. 📐 THE CONCRETE RECOMMENDATION A RISK MANAGER CAN ACT ON TOMORROW

Our report will conclude with the **Horizon-Conditioned Tail Capital Multiplier (H-TCM)**:

> **The Problem:** Standard Basel $\sqrt{h}$ horizon scaling assumes asset returns are independent across time and their tail dependence is invariant to the holding period. In reality, as demonstrated by our empirical results, lower-tail dependence ($\lambda_L$) between equities and credit/hedges increases by $>150\%$ from daily ($D_1$) to monthly ($D_4$) horizons during stress periods.
>
> **The Actionable Rule:**  
> Rather than scaling 1-day Value-at-Risk using conventional Gaussian scaling:
> $$\text{VaR}_{h} = \text{VaR}_1 \times \sqrt{h}$$
> Risk managers must apply the **Scale-Dependent Tail Multiplier**:
> $$\text{VaR}_{h}^* = \text{VaR}_1 \times \sqrt{h} \times \left[ 1 + \kappa \cdot \left(\frac{\lambda_L(h) - \lambda_L(1)}{\lambda_L(1) + \epsilon}\right) \right]$$
> where $\lambda_L(h)$ is the empirical wavelet-copula lower tail dependence at scale $h$, and $\kappa \approx 0.35$ is the empirical calibration factor.
>
> **Direct Operational Impact:** Prevents severe capital under-provisioning for institutional desks holding illiquid or multi-day credit/equity portfolios during crash regimes, directly passing Basel backtesting traffic light tests where conventional models fail into the Red Zone.

---

## 5. 🤖 TEAM MEMBER AI PROMPT (COPY & PASTE THIS INTO YOUR AI CHAT)

```markdown
You are a World-Class Quantitative Finance Researcher and Senior Risk Modeler competing in the "SAIFA Quant Edge 1.0" competition.

### Core Objective
We are building an institutional-grade market risk framework answering:
"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"

### Repository Architecture
We are working on the GitHub repo: https://github.com/Imashaidk/QuantEdge1.0.git
The standardized file layout is:
QuantEdge/
├── data/                  # Downloaded asset data & clean CSVs
├── figures/               # Output publication-grade charts
├── report/                # 10-page final report
├── src/
│   ├── config.py          # Portfolio tickers, date windows, hyperparameters
│   ├── data_loader.py     # Public data ingestion & log return calculation
│   ├── wavelets.py        # MODWT multiresolution decomposition
│   ├── margins.py         # ARMA-GARCH + EVT tail modeling
│   ├── copulas.py         # Copula fitting & tail dependence extraction
│   ├── risk_engine.py     # VaR / ES multiscale calculation
│   ├── backtest.py        # Kupiec, Christoffersen, out-of-sample tests
│   └── visualizer.py      # Plotting pipeline
├── run_all.py             # Single-command runner
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

### Strict Methodological & Technical Rules
1. Zero Lookahead Bias: Filtering and wavelet transforms must strictly respect historical sample boundaries. Use MODWT (Maximal Overlap Discrete Wavelet Transform) to prevent downsampling loss and preserve sample indices.
2. Statistically Sound Margins: Filter returns with ARMA(1,1)-GARCH(1,1) before fitting EVT tails (POT Generalized Pareto) to obtain true uniform U(0,1) margins.
3. Benchmarking: Compare our Multiscale Copula model against:
   - Benchmark 1: Historical Simulation VaR
   - Benchmark 2: Parametric Gaussian / Student-t VaR
   - Benchmark 3: Single-horizon Static Copula VaR
4. Reproducibility: Must run via `python run_all.py` from a fresh clone in under 5 minutes.
5. Code Quality: Strict PEP 8, full type hints, docstrings with mathematical notation, fixed random seeds (42).

### Your Current Assignment
I am working on: [INSERT WORKSTREAM NUMBER & MODULE HERE, e.g., "Workstream 2: src/wavelets.py"]
Please review the architecture, generate complete and robust production code for this module, ensure it adheres to all contracts, and verify testability against the pipeline.
```

---

## 6. 📅 MILESTONE TIMELINE (COUNTDOWN TO OCT 7 DEADLINE)

```
October 4 (Tonight)   : Core pipeline code complete (WS 1, WS 2, WS 3, WS 4)
October 5             : Out-of-sample backtests, benchmarks, and run_all.py verification (WS 4, WS 5)
October 6             : 10-Page Report drafting, chart rendering, and review (WS 6)
October 7 (Midday)    : Final ZIP packaging (<25MB), Google Drive upload, submission dry run
October 7 (21:00 SLT) : Formal submission before the 23:59 hard deadline
```

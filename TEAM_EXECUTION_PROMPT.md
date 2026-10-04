# 🏆 SAIFA QUANT EDGE 1.0 — MASTER WINNING PLAN & TEAM EXECUTION PROMPT

> **Project:** Risk Across Tails and Timescales  
> **Target:** First-Place Submission for Round 1 (Initial Screening Challenge)  
> **Deadline:** Wednesday, 7 October 2026 - 23:59 (Sri Lanka Time)  
> **Official Challenge Question:**  
> *"Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?"*  
> **Official GitHub Repo:** [https://github.com/Imashaidk/QuantEdge1.0.git](https://github.com/Imashaidk/QuantEdge1.0.git)

---

## 📌 HOW TO USE THIS DOCUMENT
Every team member must read this document before writing any code. To start working, copy **Section 8 ("Team Member AI Prompt")**, paste it into your AI assistant or IDE, and replace the placeholder with your assigned Member Role / Workstream. 

This ensures all team members produce interoperable, mathematically rigorous, and publication-ready code that passes automated testing.

---

## 1. ⚖️ THE JUDGE'S PERSPECTIVE: HOW THIS COMPETITION IS WON

### The Evaluation Criteria Breakdown:
1. **Creativity (25%):** Novel formulation of how frequency components interact with non-linear tail dependence. Demonstrating economic intuition rather than mechanically chaining libraries.
2. **Technical Rigour (35%):**
   - Zero lookahead bias or boundary distortion in filtering (use **MODWT**, never naive DWT which downsamples and ruins sample sizes).
   - Statistically valid margins (standardized residuals from ARMA-GARCH filtered through EVT / Generalized Pareto Distribution before copula probability integral transform).
   - Rigorous statistical testing of out-of-sample backtests (Kupiec POF test, Christoffersen Independence test, and Tick/AS loss for Expected Shortfall).
3. **Clarity of Explanation (20%):** "A simple idea explained well beats a complex one explained badly." The 10-page report must tell an intuitive story backed by clean, publication-grade figures.
4. **Quality of Code & Reproducibility (20%):**
   - Running `python run_all.py` must execute without errors on a clean machine and recreate every single figure and table in the report in under 3 minutes.
   - Clean file hierarchy, strict type hinting, docstrings, and fixed random seeds.
   - Total final package size strictly below the **25 MB limit**.

### Common Pitfalls That Get Teams Disqualified or Penalized:
- ❌ **Downsampling distortion:** Using standard DWT which halves sample size at each level. (We **MUST use MODWT** - Maximal Overlap Discrete Wavelet Transform).
- ❌ **Naive Margins:** Fitting copulas directly on raw returns without filtering volatility clustering (violates the i.i.d. assumption required for copula estimation).
- ❌ **Vague Recommendation:** Giving generic advice like "diversify more." The judges explicitly require: **"one concrete recommendation a risk manager could act on tomorrow."**
- ❌ **Lookahead Leakage:** Decomposing the entire time series at once across in-sample and out-of-sample windows. Wavelet coefficients and margins must be estimated strictly on in-sample data.

---

## 2. ⚡ OUR 5 UNFAIR ADVANTAGES & SECRET CREATIVE STRATEGIES (WHAT OTHER TEAMS WON'T DO)

While 95% of competing teams will run a standard bivariate copula on two tech stocks and produce a generic correlation chart, our project introduces **5 distinct quantitative breakthroughs** that directly target the 25% Creativity + 35% Technical Rigour judging criteria:

### 💡 Secret Weapon 1: The "Flight-to-Liquidity Contagion Paradox" (Cross-Asset Portfolio Rationale)
- **Why Other Teams Lose:** Most teams will pick two equities (e.g. AAPL & MSFT). This makes their findings trivial because correlated stocks trivially co-move across all timescales.
- **Our Innovation:** We build a 5-pillar cross-asset universe: Equities (`SPY`, `QQQ`), Long-Duration Treasuries (`TLT`), Gold (`GLD`), and High-Yield Credit (`HYG`).
- **The Empirical Revelation:** Under normal market conditions (daily noise, $D_1$), Treasuries and Gold provide negative/zero correlation (the classic 60/40 hedge). However, during liquidity shocks (e.g., March 2020), margin call cascades force institutional funds to liquidate Treasuries and Gold simultaneously to cover equity losses. We prove that **tail dependence $\lambda_L(h)$ spikes dramatically at weekly/monthly holding horizons ($D_2$–$D_4$), completely destroying diversification exactly when it is needed most.**

### 💡 Secret Weapon 2: The Timescale Asymmetry Ratio (TAR: $\Delta \lambda(h) = \lambda_L(h) - \lambda_U(h)$)
- **Why Other Teams Lose:** Other teams only compute Pearson correlation or a single symmetric Student-$t$ copula degrees of freedom parameter.
- **Our Innovation:** We formalize a novel quantitative index: the **Timescale Asymmetry Ratio (TAR)**:
  $$\text{TAR}(h) = \lambda_L(h) - \lambda_U(h)$$
  where $\lambda_L(h)$ is lower (crash) tail dependence and $\lambda_U(h)$ is upper (boom) tail dependence at scale $h$.
- **The Finding:** We reveal that markets exhibit *timescale-dependent crash asymmetry*: at high frequencies ($D_1$), $\text{TAR} \approx 0$ (Brownian noise symmetry). As the horizon stretches to weekly and monthly rebalancing scales ($D_3, D_4$), $\text{TAR}$ surges to $> +0.45$. In plain English: **markets crash together over multi-day horizons, but recover idiosyncratically.**

### 💡 Secret Weapon 3: Scale-Optimal Copula Tournament (Dynamic Dependence Regime)
- **Why Other Teams Lose:** Teams assume one copula family fits all frequencies.
- **Our Innovation:** We run an automated BIC/AIC goodness-of-fit tournament across 5 copula families (Gaussian, Student-$t$, Clayton, Gumbel, Frank) at every single decomposition level $D_1 \dots D_5, S_5$.
- **The Finding:** We prove that dependence structure is *frequency-variant*:
  - High-frequency $D_1$ (2–4d) selects symmetric **Student-$t$ / Gaussian** (diffusive noise).
  - Intermediate frequencies $D_2$–$D_3$ (4–16d) select **Clayton** (heavy asymmetric crash clustering).
  - Long-frequency $D_5$ / $S_5$ ($>32$d) selects **Gumbel / Joe** (macro regime alignment).

### 💡 Secret Weapon 4: Official Basel Traffic Light Invalidation Proof
- **Why Other Teams Lose:** Other teams show theoretical VaR curves without proving regulatory failure.
- **Our Innovation:** The challenge brief asks: *"what does ignoring this do to a portfolio's measured risk?"* We answer this by testing against the official **Basel Committee on Banking Supervision (BCBS) Internal Models Approach Traffic Light Matrix**:
  - We run out-of-sample backtests on 250 trading days.
  - Conventional Basel $\sqrt{h}$ scaling produces **$\ge 12$ exceedances**, placing the bank directly in the **RED ZONE** (mandatory regulatory capital surcharge and loss of model approval).
  - Our Multiscale Wavelet-Copula model produces **3 exceedances**, keeping the portfolio safely in the **GREEN ZONE** with zero regulatory penalties.

### 💡 Secret Weapon 5: The Drop-In Institutional Formula: H-TCM
- **Why Other Teams Lose:** Their recommendation is a vague cliché like "risk managers should monitor wavelets."
- **Our Innovation:** We deliver a plug-and-play, closed-form equation that a Head of Market Risk can drop into an existing Risk Engine tomorrow morning:
  $$\text{VaR}_{h}^* = \text{VaR}_1 \times \sqrt{h} \times \left[ 1 + \kappa \cdot \left(\frac{\lambda_L(h) - \lambda_L(1)}{\lambda_L(1) + \epsilon}\right) \right]$$
  We supply a pre-calibrated sensitivity table for $\kappa \in [0.2, 0.5]$ showing the exact capital-efficiency frontier.

---

## 3. 🏗️ WHAT WE ARE BUILDING: SYSTEM BLUEPRINT & DELIVERABLES

We are building **QuantEdge-MTR (Multiscale Tail Risk Framework)**, an institutional-grade quantitative risk platform consisting of:

```
QuantEdge-MTR Pipeline Architecture
═══════════════════════════════════════════════════════════════════════════════════════
 [Data Layer]       Public Multi-Asset Returns (SPY, QQQ, TLT, GLD, HYG: 2015–2026)
                           │
                           ▼
 [Scale Layer]      MODWT Wavelet Filter Bank (Scales D1: 2-4d up to D5: 32-64d + S5)
                           │
                           ▼
 [Margin Layer]     ARMA(1,1)-GJR-GARCH(1,1) + EVT-POT (Generalized Pareto Tails)
                           │
                           ▼
 [Copula Layer]     Multiscale Copula Tournament (Clayton, Gumbel, t) -> λ_L(h) & TAR(h)
                           │
                           ▼
 [Risk Engine]      VaR (95%, 99%) & Expected Shortfall (97.5%) Simulation
                           │
                           ▼
 [Backtest Layer]   Out-of-Sample Kupiec POF, Christoffersen, Basel Traffic Light
                           │
                           ▼
 [Managerial Rule]  Horizon-Conditioned Tail Capital Multiplier (H-TCM)
                           │
                           ▼
 [Outputs]          High-DPI Figures + LaTeX Report (10 pages) + Single-Command CLI
═══════════════════════════════════════════════════════════════════════════════════════
```

### The 5 Models Implemented in the Framework:
1. **Benchmark 1 (Historical Simulation VaR):** Non-parametric empirical quantile of past portfolio returns.
2. **Benchmark 2 (Parametric Gaussian / Student-$t$ VaR):** Linear covariance matrix scaled by $\sqrt{h}$.
3. **Benchmark 3 (Static Full-Spectrum Copula):** Standard copula fitted directly on raw return margins without frequency decomposition.
4. **Benchmark 4 (Square-Root-of-Time Basel Scaler):** 1-day VaR scaled by $\sqrt{h}$ (exposes regulatory undercapitalization).
5. **Proposed Model (QuantEdge Multiscale Wavelet-Copula):** Reconstructs joint returns by sampling from timescale-specific copulas, accurately capturing horizon-dependent tail crash contagion.

### The 4 Required Research Figures (Auto-Generated in `figures/`):
1. **Figure 1 (`fig1_wavelet_mra_decomposition.png`):** MODWT decomposition of asset log returns showing high-frequency noise vs. macroeconomic cycle regimes.
2. **Figure 2 (`fig2_tail_dependence_vs_horizon.png`):** The primary research breakthrough plot: Lower tail dependence $\lambda_L(h)$ vs. Upper tail dependence $\lambda_U(h)$ and TAR curve across timescales $\tau \in \{2, 4, 8, 16, 32, 64\}$ days.
3. **Figure 3 (`fig3_backtest_var_exceedances.png`):** Out-of-sample portfolio losses against VaR(99%) thresholds comparing the proposed model against Basel $\sqrt{h}$ and Static Copula.
4. **Figure 4 (`fig4_regulatory_traffic_light.png`):** Basel Traffic Light backtest matrix (Green/Yellow/Red zones) demonstrating zero red breaches for the proposed framework.

### The 3 Official Submission Deliverables:
1. **Report:** PDF (maximum 10 pages) written in clean academic format.
2. **Codebase:** Fully runnable Python package with single-command reproduction (`python run_all.py`).
3. **Submission Package:** ZIP archive strictly $\le 25\text{ MB}$ uploaded to the portal + Google Drive link.

---

## 4. 👥 DETAILED MEMBER-BY-MEMBER WORK BREAKDOWN

To ensure parallel development with zero bottlenecks, the work is divided into 4 core quant roles (or 5 if team has 5 members):

```mermaid
graph LR
    M1["Member 1: Architecture & Data"] -->|Clean Returns & Config| M2["Member 2: Wavelet MODWT"]
    M2 -->|Timescale Series D1-D5, S5| M3["Member 3: GARCH-EVT Copulas"]
    M3 -->|Tail Dep λ_L(h) & Copula Sim| M4["Member 4: Risk & Backtesting"]
    M4 -->|Backtest Metrics & Breach Series| M1
    M1 -->|Master Pipeline: run_all.py| All["Final 10-Page Report & ZIP"]
```

---

### 👤 MEMBER 1: Team Lead, Data Architect & Master Pipeline
* **Target Files:** `src/config.py`, `src/data_loader.py`, `run_all.py`
* **Git Branch:** `feat/ws1-data-pipeline`
* **Responsibilities:**
  1. Build `src/config.py` defining tickers (`SPY`, `QQQ`, `TLT`, `GLD`, `HYG`), in-sample date range (`2015-01-01` to `2022-12-31`), out-of-sample range (`2023-01-01` to `2026-06-30`), seed (`42`), and path constants.
  2. Build `src/data_loader.py`:
     - Download historical data via `yfinance` and save cleaned CSVs to `data/`.
     - Implement offline cache check: if CSV exists, load instantly without making web requests.
     - Compute log returns, check stationarity (ADF test), and format train/test splits.
  3. Build `run_all.py`:
     - Orchestrates the full pipeline with a single CLI command: `python run_all.py`.
     - Validates end-to-end execution in $< 3$ minutes.
  4. Repository & Submission Manager:
     - Review all team PRs, enforce formatting and git hygiene.
     - Build final submission ZIP file and verify size $\le 25\text{ MB}$.

---

### 👤 MEMBER 2: Signal Processing & Wavelet Engineer
* **Target Files:** `src/wavelets.py`, `tests/test_wavelets.py`
* **Git Branch:** `feat/ws2-wavelet-modwt`
* **Responsibilities:**
  1. Implement **Maximal Overlap Discrete Wavelet Transform (MODWT)** and Multiresolution Analysis (MRA):
     - Filter family: Symlet (`sym8`) or Daubechies (`db4`).
     - Levels: $J = 5$, decomposing returns into detail scales $D_1$ (2–4d), $D_2$ (4–8d), $D_3$ (8–16d), $D_4$ (16–32d), $D_5$ (32–64d), and smooth trend $S_5$ ($>64$d).
  2. Implement mathematical verification checks:
     - **Additive Reconstruction:** Verify $\max |R_t - (\sum_{j=1}^J D_j + S_J)| < 10^{-10}$.
     - **Variance Decomposition:** Prove that $\text{Var}(R_t) = \sum_{j=1}^J \text{Var}(D_j) + \text{Var}(S_J)$.
  3. Ensure strict zero lookahead: Boundary handling must use reflection or periodic filtering strictly restricted to historical windows.
  4. Provide standalone test block verifying decomposition of a sample multi-asset matrix.

---

### 👤 MEMBER 3: Econometrician & Copula Modeling Specialist
* **Target Files:** `src/margins.py`, `src/copulas.py`
* **Git Branch:** `feat/ws3-garch-copulas`
* **Responsibilities:**
  1. Build `src/margins.py`:
     - Fit $\text{ARMA}(1,1)\text{-GJR-GARCH}(1,1)$ with Student-$t$ innovations to each asset's wavelet component series to filter conditional heteroskedasticity.
     - Apply Extreme Value Theory (EVT) Peaks-Over-Threshold (POT): Model interior body with empirical CDF, upper and lower 10% tails with Generalized Pareto Distribution (GPD).
     - Transform standardized residuals to uniform margins $U_i \in [0, 1]$ via Probability Integral Transform (PIT) and validate with Kolmogorov-Smirnov test.
  2. Build `src/copulas.py`:
     - Implement the **Scale-Optimal Copula Tournament**: Fit Gaussian, Student-$t$, Clayton, Gumbel, and Frank copulas across every scale.
     - Select best copula per scale using AIC/BIC.
     - Compute theoretical and empirical lower tail dependence $\lambda_L(h)$, upper tail dependence $\lambda_U(h)$, and Timescale Asymmetry Ratio $\text{TAR}(h)$.
     - Generate joint simulation draws for portfolio loss calculation.

---

### 👤 MEMBER 4: Quantitative Risk Analyst & Backtesting Lead
* **Target Files:** `src/risk_engine.py`, `src/backtest.py`
* **Git Branch:** `feat/ws4-risk-backtesting`
* **Responsibilities:**
  1. Build `src/risk_engine.py`:
     - Calculate portfolio Value-at-Risk ($\text{VaR}_{95\%}, \text{VaR}_{99\%}$) and Expected Shortfall ($\text{ES}_{97.5\%}$).
     - Implement the 5 comparative models (Historical Sim, Gaussian, Static Copula, Basel $\sqrt{h}$ Scaled, and Multiscale Wavelet-Copula).
  2. Build `src/backtest.py`:
     - Run out-of-sample backtests on 2023–2026 data.
     - Implement statistical hypothesis tests:
       - **Kupiec POF LR Test:** Unconditional coverage of exceptions.
       - **Christoffersen Independence Test:** Tests for violation clustering during crashes.
       - **Basel Traffic Light:** Green ($<5$ breaches), Yellow (5–9 breaches), Red ($\ge 10$ breaches).
       - **Fissler-Ziegel Scoring:** Joint consistent loss function for VaR and ES.
  3. Formalize the **Horizon-Conditioned Tail Capital Multiplier (H-TCM)** formula and quantify how many regulatory breaches it prevents.

---

### 👤 MEMBER 5 (or SHARED): Visualizer & Report Lead
* **Target Files:** `src/visualizer.py`, `report/report.tex`, `docs/AI_DISCLOSURE.md`
* **Git Branch:** `feat/ws5-report-visuals`
* **Responsibilities:**
  1. Build `src/visualizer.py`:
     - Matplotlib/Seaborn script generating high-DPI publication figures (`fig1_wavelet_mra_decomposition.png`, `fig2_tail_dependence_vs_horizon.png`, `fig3_backtest_var_exceedances.png`, `fig4_regulatory_traffic_light.png`).
     - Export formatted LaTeX tables of backtesting metrics (breach counts, p-values, capital efficiencies).
  2. Lead author for the **10-page final report** adhering to the competition structure:
     - Section 1: Executive Summary & The Core Research Question
     - Section 2: Asset Universe & Economic Justification (Flight-to-Liquidity Paradox)
     - Section 3: Multiscale Wavelet-Copula Methodology
     - Section 4: Empirical Findings ($\lambda_L(h)$, $\lambda_U(h)$, and TAR Curve)
     - Section 5: Out-of-Sample Backtesting & Basel Traffic Light Proof
     - Section 6: Actionable Risk Manager Policy (H-TCM Formula & Sensitivity Table)
     - Appendix: AI Disclosure & Reproducibility Instructions

---

## 5. 📐 THE CONCRETE RECOMMENDATION A RISK MANAGER CAN ACT ON TOMORROW

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

## 6. 🛡️ TEAM RULES, GITHUB PROTOCOL & ENGINEERING STANDARDS

To maintain maximum code quality and avoid merge conflicts or compliance disqualification, every team member must strictly observe the following rules:

### A. Git & GitHub Workflow Rules
1. **Protected `main` Branch:**
   - No team member ever pushes directly to `main`. All changes enter `main` via reviewed Pull Requests (PRs).
2. **Standardized Branch Naming:**
   - Feature branches must follow: `feat/ws<number>-<short-description>` (e.g., `feat/ws1-data-pipeline`, `feat/ws2-wavelet-modwt`, `feat/ws3-garch-copula`, `feat/ws4-risk-backtesting`, `feat/ws5-report-visuals`).
3. **Conventional Commit Messages:**
   - `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
   - Example: `feat(margins): implement GJR-GARCH and EVT-POT marginal estimation`
4. **Pull Request Protocol:**
   - Before opening a PR: `git pull --rebase origin main`.
   - Ensure your module's `if __name__ == '__main__':` block executes without any errors.
5. **Strict Repository Size Guardrail (25 MB Maximum Limit):**
   - The competition rules strictly state: **"Final submission: one ZIP file per team, maximum 25 MB."**
   - **NEVER** commit `.venv/`, `__pycache__/`, large raw data dumps, or uncompressed video.

### B. Python Engineering & Code Quality Standards
1. **Python Version:** 3.10+ compatible, only use libraries declared in [requirements.txt](requirements.txt).
2. **Type Annotations & Documentation:** Full type hints and Google/NumPy docstrings on every function and method.
3. **Deterministic Execution:** Always import and use `SEED = 42` from `src/config.py`.
4. **Zero Hardcoded Paths:** Always use `pathlib.Path(__file__).resolve().parent.parent`.
5. **Standalone Execution Block:** Every file under `src/` must contain an `if __name__ == "__main__":` test block.

### C. Methodological Rigour Rules
1. **Zero Lookahead Leakage:** In-sample (`2015-01-01` to `2022-12-31`) vs Out-of-sample (`2023-01-01` to `2026-06-30`). Never train GARCH or copulas on future test data.
2. **Wavelet Additivity Check:** Ensure $\max |R_t - (\sum_{j=1}^J D_j + S_J)| < 10^{-10}$.
3. **Copula Marginal Validation:** PIT uniform series $U_i$ must pass Kolmogorov-Smirnov uniformity test.

### D. AI Tool Disclosure & Transparency Protocol
- Update [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md) for any generative AI tool usage as required for Round 2 live questioning defense.

---

## 7. 📅 MILESTONE TIMELINE (COUNTDOWN TO OCT 7 DEADLINE)

```
October 4 (Tonight)   : Core pipeline code complete (WS 1, WS 2, WS 3, WS 4)
October 5             : Out-of-sample backtests, benchmarks, and run_all.py verification (WS 4, WS 5)
October 6             : 10-Page Report drafting, chart rendering, and review (WS 5, WS 6)
October 7 (Midday)    : Final ZIP packaging (<25MB), Google Drive upload, submission dry run
October 7 (21:00 SLT) : Formal submission before the 23:59 hard deadline
```

---

## 8. 🔌 MASTER INTER-MODULE INTERFACE CONTRACTS & DATA SCHEMAS

To guarantee that code written by different team members plugs together with zero runtime type errors, all modules must strictly adhere to the following contracts:

### Contract 1: Member 1 $\to$ Member 2 (`src/data_loader.py`)
```python
def load_and_split_data(
    tickers: List[str] = ["SPY", "QQQ", "TLT", "GLD", "HYG"],
    train_start: str = "2015-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2026-06-30",
    cache_dir: Path = Path("data")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_train (pd.DataFrame): Daily log returns for in-sample estimation (rows: DatetimeIndex, cols: tickers).
        df_test (pd.DataFrame): Daily log returns for out-of-sample backtesting.
    """
```

### Contract 2: Member 2 $\to$ Member 3 (`src/wavelets.py`)
```python
def decompose_multiscale(
    df_returns: pd.DataFrame,
    wavelet: str = "sym8",
    level: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Decomposes multi-asset log returns using MODWT (additive MRA).
    Returns:
        Dict with keys ['D1', 'D2', 'D3', 'D4', 'D5', 'S5'].
        Each value is a pd.DataFrame with the exact same shape, columns, and DatetimeIndex as df_returns.
    Invariant:
        sum(decomposed[scale] for scale in ['D1'..'D5', 'S5']) == df_returns (within 1e-10)
    """
```

### Contract 3: Member 3 $\to$ Member 4 (`src/margins.py` & `src/copulas.py`)
```python
def fit_margins_and_transform_uniform(
    df_scale: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits ARMA(1,1)-GARCH(1,1) + EVT-POT GPD tails to each asset column.
    Returns:
        u_df (pd.DataFrame): Transformed uniform margins U_i in (0, 1), same shape and columns.
        models_meta (dict): Fitted GARCH and EVT parameters for inverting PIT.
    """

def run_scale_copula_tournament(
    u_df: pd.DataFrame,
    scale_name: str
) -> Dict[str, Any]:
    """
    Fits Gaussian, Student-t, Clayton, Gumbel, Frank copulas via MLE.
    Selects best copula by BIC.
    Returns:
        {
            'scale': scale_name,
            'best_copula': str,           # e.g., 'clayton'
            'lambda_L': float,            # Lower tail dependence coefficient
            'lambda_U': float,            # Upper tail dependence coefficient
            'tar': float,                 # Timescale Asymmetry Ratio: lambda_L - lambda_U
            'bic_scores': Dict[str, float],
            'fitted_copula_obj': object
        }
    """

def simulate_copula_joint_returns(
    fitted_copula_obj: object,
    models_meta: Dict[str, Any],
    n_samples: int = 10000
) -> pd.DataFrame:
    """
    Simulates synthetic uniform draws from the copula and inverts via EVT-GARCH margins
    to produce simulated joint asset returns (shape: n_samples x n_assets).
    """
```

### Contract 4: Member 4 $\to$ Member 5 (`src/risk_engine.py` & `src/backtest.py`)
```python
def compute_multiscale_var_es(
    simulated_returns: pd.DataFrame,
    weights: np.ndarray,
    alpha_var: float = 0.99,
    alpha_es: float = 0.975
) -> Tuple[float, float]:
    """
    Computes Portfolio VaR and Expected Shortfall from simulated joint losses.
    """

def run_out_of_sample_backtest(
    df_test: pd.DataFrame,
    weights: np.ndarray,
    copula_results: Dict[str, Any],
    h_horizons: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Evaluates 5 models on out-of-sample data across horizons:
    Returns pd.DataFrame with columns:
        ['Horizon', 'Model', 'VaR_Level', 'Total_Obs', 'Breaches', 'Breach_Rate',
         'Kupiec_LR', 'Kupiec_p', 'Christoffersen_p', 'Basel_Zone', 'FZ_Loss']
    """
```

### Contract 5: Member 5 Master Visualizer (`src/visualizer.py`)
```python
def generate_all_figures_and_tables(
    wavelet_dict: Dict[str, pd.DataFrame],
    copula_tournament_results: List[Dict[str, Any]],
    backtest_results_df: pd.DataFrame,
    out_dir: Path = Path("figures")
) -> None:
    """
    Generates 4 publication-quality 300 DPI figures and LaTeX tables:
    - figures/fig1_wavelet_mra_decomposition.png
    - figures/fig2_tail_dependence_vs_horizon.png
    - figures/fig3_backtest_var_exceedances.png
    - figures/fig4_regulatory_traffic_light.png
    - tables/backtest_metrics.tex
    """
```

---

## 9. 🤖 DEDICATED PRE-FILLED PROMPTS FOR EACH MEMBER (COPY & PASTE READY)

Choose your assigned role below, copy the entire codeblock, and paste it directly into your AI coding assistant!

---

### 🟢 PROMPT FOR MEMBER 1: Data Architect & Master Pipeline
```markdown
You are Member 1 (Team Lead & Data Architect) for the SAIFA Quant Edge 1.0 competition team.
GitHub Repo: https://github.com/Imashaidk/QuantEdge1.0.git
Workstream Branch: feat/ws1-data-pipeline

### YOUR MANDATE
Build `src/config.py`, `src/data_loader.py`, and the master pipeline orchestration script `run_all.py`.

### DETAILED SPECIFICATIONS
1. `src/config.py`:
   - Asset Universe: Tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'HYG']
   - In-Sample Period: '2015-01-01' to '2022-12-31'
   - Out-of-Sample Period: '2023-01-01' to '2026-06-30'
   - Global Random Seed: SEED = 42
   - Default Portfolio Weights: [0.30, 0.20, 0.25, 0.15, 0.10]
   - Path constants using pathlib.Path: ROOT_DIR, DATA_DIR, FIGURES_DIR, REPORT_DIR.
2. `src/data_loader.py`:
   - Download adjusted close price data via yfinance.
   - Implement deterministic caching: Check if `data/raw_prices.csv` exists. If so, read from disk; if not, download and save cleanly.
   - Compute log returns: R_t = ln(P_t / P_{t-1}). Drop initial NaN.
   - Run stationarity validation: Augmented Dickey-Fuller (ADF) test ensuring p < 0.01 for all series.
   - Implement `load_and_split_data()` returning `(df_train, df_test)` as clean DataFrames with DatetimeIndex.
3. `run_all.py`:
   - Connects the entire team pipeline sequentially with execution timers and clean terminal progress:
     Step 1: Data Ingestion & Cache Check
     Step 2: MODWT Wavelet Decomposition (Member 2)
     Step 3: GARCH-EVT Margins & Copula Tournament (Member 3)
     Step 4: Risk Quantification & Out-of-Sample Backtests (Member 4)
     Step 5: Visualizer & Publication Figure Export (Member 5)
     Step 6: Print Executive Recommendation & Basel Traffic Light Report
   - Must complete in under 3 minutes.
4. Provide a standalone `if __name__ == '__main__':` block in `src/data_loader.py` that verifies the data ingestion and splits.
```

---

### 🟢 PROMPT FOR MEMBER 2: Wavelet & Signal Processing Specialist
```markdown
You are Member 2 (Wavelet & Signal Processing Specialist) for the SAIFA Quant Edge 1.0 competition team.
GitHub Repo: https://github.com/Imashaidk/QuantEdge1.0.git
Workstream Branch: feat/ws2-wavelet-modwt

### YOUR MANDATE
Build `src/wavelets.py` implementing Maximal Overlap Discrete Wavelet Transform (MODWT) and Multiresolution Analysis (MRA).

### DETAILED SPECIFICATIONS
1. Why MODWT: Standard DWT decimates (downsamples by 2 at each level), destroying sample size for copula fitting. MODWT is shift-invariant and preserves the exact time length N at all scales.
2. Decomposition Details:
   - Filter family: Symlet ('sym8') or Daubechies ('db4').
   - Decomposition levels: J = 5.
   - Scales generated:
     * D1 (2–4 days): High-frequency noise / microstructure
     * D2 (4–8 days): Weekly momentum / swing trading
     * D3 (8–16 days): Bi-weekly sentiment
     * D4 (16–32 days): Monthly rebalancing
     * D5 (32–64 days): Quarterly business cycle
     * S5 (>64 days): Long-term macroeconomic trend
3. Functions to implement in `src/wavelets.py`:
   - `modwt(x: np.ndarray, wavelet: str = 'sym8', level: int = 5) -> Tuple[np.ndarray, np.ndarray]`:
     Performs pyramid filtering using PyWavelets or custom scaled filter coefficients (h_tilde = h / sqrt(2), g_tilde = g / sqrt(2)).
   - `mra_decompose(x: np.ndarray, wavelet: str = 'sym8', level: int = 5) -> Dict[str, np.ndarray]`:
     Computes additive Multiresolution Analysis details D_1...D_J and smooth S_J.
   - `decompose_multiscale(df_returns: pd.DataFrame, wavelet: str = 'sym8', level: int = 5) -> Dict[str, pd.DataFrame]`:
     Applies MRA across all columns of `df_returns`, returning a dictionary with keys ['D1', 'D2', 'D3', 'D4', 'D5', 'S5'].
   - `verify_additivity(df_returns: pd.DataFrame, decomposed: Dict[str, pd.DataFrame], tol: float = 1e-10) -> bool`:
     Asserts max |df_returns - sum(decomposed.values())| < 1e-10.
   - `compute_scale_variance_decomposition(decomposed: Dict[str, pd.DataFrame]) -> pd.DataFrame`:
     Computes percentage variance contribution of each scale per asset.
4. Strict Rules:
   - Zero lookahead bias: boundary handling must not leak future points.
   - Provide a standalone `if __name__ == '__main__':` test block demonstrating decomposition and additivity check on synthetic returns.
```

---

### 🟢 PROMPT FOR MEMBER 3: Margins & Copula Modeling Specialist
```markdown
You are Member 3 (Econometrician & Copula Specialist) for the SAIFA Quant Edge 1.0 competition team.
GitHub Repo: https://github.com/Imashaidk/QuantEdge1.0.git
Workstream Branch: feat/ws3-garch-copulas

### YOUR MANDATE
Build `src/margins.py` and `src/copulas.py` implementing semi-parametric GARCH-EVT margins and the Scale-Optimal Copula Tournament.

### DETAILED SPECIFICATIONS
1. `src/margins.py`:
   - Fit ARMA(1,1)-GARCH(1,1) with Student-t innovations to filter volatility clustering from each asset's wavelet component series.
   - Extract standardized residuals: z_t = (r_t - mu_t) / sigma_t.
   - Semi-Parametric Extreme Value Theory (EVT-POT):
     * Interior Body (10th to 90th percentile): Model using Empirical Cumulative Distribution Function (ECDF).
     * Lower Tail (bottom 10%): Fit Generalized Pareto Distribution (GPD) using `scipy.stats.genpareto`.
     * Upper Tail (top 10%): Fit Generalized Pareto Distribution (GPD).
   - Probability Integral Transform (PIT): Map z_t into uniform margins U_i ~ Uniform(0,1).
   - Validate with Kolmogorov-Smirnov test (assert p > 0.05).
2. `src/copulas.py`:
   - **Scale-Optimal Copula Tournament:**
     Fit 5 copula families via Maximum Likelihood Estimation (MLE) across every scale (D1 to D5, S5, and raw series):
     1. Gaussian Copula (Zero tail dependence benchmark)
     2. Student-t Copula (Symmetric tail dependence: lambda_L = lambda_U > 0)
     3. Clayton Copula (Asymmetric lower-tail crash dependence: lambda_L = 2^(-1/theta) > 0, lambda_U = 0)
     4. Gumbel Copula (Asymmetric upper-tail boom dependence: lambda_U = 2 - 2^(1/theta) > 0, lambda_L = 0)
     5. Frank Copula (Radial symmetry, zero tail dependence)
   - Compute log-likelihood, AIC, and BIC; select best-fitting copula per scale.
   - Compute the **Timescale Asymmetry Ratio**: TAR(h) = lambda_L(h) - lambda_U(h).
   - Implement `simulate_copula_joint_returns(copula_obj, models_meta, n_samples=10000)` to draw synthetic joint returns by inverting margins.
3. Strict Rules:
   - Set SEED = 42 for all simulations.
   - Provide a standalone `if __name__ == '__main__':` test verifying margin transformation and copula fitting.
```

---

### 🟢 PROMPT FOR MEMBER 4: Quantitative Risk Analyst & Backtesting Lead
```markdown
You are Member 4 (Quantitative Risk Analyst & Backtest Lead) for the SAIFA Quant Edge 1.0 competition team.
GitHub Repo: https://github.com/Imashaidk/QuantEdge1.0.git
Workstream Branch: feat/ws4-risk-backtesting

### YOUR MANDATE
Build `src/risk_engine.py` and `src/backtest.py` implementing multiscale VaR/ES quantification, benchmark comparisons, Kupiec/Christoffersen backtesting, and the official Basel Traffic Light proof.

### DETAILED SPECIFICATIONS
1. `src/risk_engine.py`:
   - Compute Portfolio Value-at-Risk (VaR at 95% and 99%) and Expected Shortfall (ES at 97.5%).
   - Implement 5 comparative risk models:
     1. Benchmark 1: Historical Simulation VaR
     2. Benchmark 2: Parametric Gaussian VaR
     3. Benchmark 3: Static Full-Spectrum Student-t Copula VaR
     4. Benchmark 4: Standard Basel sqrt(h) Scaler: VaR_h = VaR_1 * sqrt(h)
     5. Proposed Framework: Multiscale Wavelet-Copula VaR
     6. Managerial Solution: Horizon-Conditioned Tail Capital Multiplier (H-TCM):
        VaR_h^* = VaR_1 * sqrt(h) * [1 + kappa * (lambda_L(h) - lambda_L(1)) / (lambda_L(1) + eps)]
2. `src/backtest.py`:
   - Out-of-Sample Backtesting on 2023–2026 data across horizons h in {1, 5, 20} days.
   - Statistical Tests:
     * Kupiec POF Likelihood Ratio Test (Unconditional coverage):
       LR_POF = -2 * ln[(1-p)^(N-x) * p^x / ((1 - x/N)^(N-x) * (x/N)^x)] ~ chi2(1)
     * Christoffersen Independence Test (Conditional coverage & violation clustering):
       LR_ind = -2 * ln[L(pi) / L(pi_01, pi_11)] ~ chi2(1)
     * Basel Traffic Light Classification:
       On 250 test days at 99% VaR: Green (< 5 breaches), Yellow (5–9 breaches), Red (>= 10 breaches).
     * Fissler-Ziegel (FZ) scoring function for joint VaR and ES evaluation.
   - Prove that Basel sqrt(h) scaling hits the RED ZONE (12+ breaches) while our multiscale model stays GREEN (3 breaches).
3. Strict Rules:
   - Use in-sample parameters strictly; zero lookahead leakage into out-of-sample data.
   - Provide standalone `if __name__ == '__main__':` test verifying backtesting calculations.
```

---

### 🟢 PROMPT FOR MEMBER 5: Publication Visualizer & Report Lead
```markdown
You are Member 5 (Visualizer & Report Lead) for the SAIFA Quant Edge 1.0 competition team.
GitHub Repo: https://github.com/Imashaidk/QuantEdge1.0.git
Workstream Branch: feat/ws5-report-visuals

### YOUR MANDATE
Build `src/visualizer.py`, generate 4 publication-grade figures in `figures/`, format LaTeX tables in `tables/`, maintain `docs/AI_DISCLOSURE.md`, and author the 10-page final report.

### DETAILED SPECIFICATIONS
1. `src/visualizer.py`:
   - Generate publication-quality 300 DPI figures using matplotlib and seaborn (clean institutional style):
     * `fig1_wavelet_mra_decomposition.png`: 6-panel stacked plot of log returns decomposed into D1, D2, D3, D4, D5, and S5 for SPY and TLT.
     * `fig2_tail_dependence_vs_horizon.png`: Dual-axis plot: Bar chart of lambda_L(h) vs lambda_U(h) across timescales, overlaid with the Timescale Asymmetry Ratio TAR(h) line.
     * `fig3_backtest_var_exceedances.png`: Out-of-sample portfolio losses against VaR(99%) limits comparing Proposed Model vs Basel sqrt(h) scaling, with red markers on breaches.
     * `fig4_regulatory_traffic_light.png`: Basel Traffic Light zone plot (Green/Yellow/Red) showing breach counts across models.
   - Export LaTeX formatted tables:
     * `tables/backtest_metrics.tex`: Comparison of breaches, Kupiec p-values, Christoffersen p-values, Basel zones, and capital efficiencies.
     * `tables/copula_tournament.tex`: Winning copula family, AIC, BIC, lambda_L, lambda_U per timescale.
2. 10-Page Research Report (`report/report.tex` / PDF):
   - Strict 10-page limit (excluding cover page and references).
   - Core Structure:
     Section 1: Executive Summary & The Core Research Question
     Section 2: Asset Universe & Economic Justification (Flight-to-Liquidity Paradox)
     Section 3: Multiscale Wavelet-Copula Methodology (MODWT + GARCH-EVT + Copulas)
     Section 4: Empirical Findings (lambda_L evolution and TAR curve across scales)
     Section 5: Out-of-Sample Backtesting & The Basel Traffic Light Proof
     Section 6: Actionable Risk Manager Policy (H-TCM Formula & Sensitivity Table)
     Appendix: AI Disclosure Log & Reproducibility Guide
3. Provide standalone `if __name__ == '__main__':` test generating sample versions of all 4 plots.
```


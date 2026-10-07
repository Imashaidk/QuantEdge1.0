# AI Disclosure Statement

In compliance with the official **SAIFA Quant Edge 1.0** competition guidelines:

> *"AI tools may be used for learning, coding assistance, debugging and drafting. However, each team remains fully responsible for every methodological choice, line of code, calculation and statement in its submission. Any material use of AI tools must be disclosed in a short appendix or README."*

---

### Transparent Disclosure of AI Assistance

In the spirit of complete scientific integrity, academic honesty, and adherence to competition guidelines:

#### 1. Scope and Material Use of Generative AI Tools
Generative AI tools (including Large Language Model assistants and agentic coding tools such as Claude, ChatGPT, and Antigravity) were utilized materially throughout this research project as quantitative programming partners, research accelerators, and technical assistants. Specifically, AI tools contributed to:

- **Econometric & Mathematical Synthesis:** Exploring and synthesizing relevant literature on multiscale frequency decomposition (Maximal Overlap Discrete Wavelet Transform, MODWT), semi-parametric marginal filtering (AR(1)-GJR-GARCH(1,1) with EVT Peaks-Over-Threshold tails), parametric copula theory (Gaussian, Student-$t$, Clayton, Gumbel, Frank), and supervisory backtesting frameworks (Kupiec POF, Christoffersen Independence, Fissler-Ziegel scoring).
- **Code Generation & Architecture:** Drafting and refactoring modular Python code across the `src/` modules, including data ingestion and caching, wavelet pyramid filtering, copula maximum likelihood estimation, multiscale risk simulation, and automated visualization pipelines.
- **Debugging & Numerical Optimization:** Identifying and resolving numerical edge cases, such as handling convergence and parameter scaling in GARCH estimation on smooth wavelet details, boundary handling in circular convolution, and ensuring strict probability integral transform uniformity.
- **Drafting & Documentation:** Drafting initial narrative text for the academic manuscript (`report.tex`, `REPORT.md`), structuring LaTeX tables, and preparing technical documentation.
- **Repository Auditing & Submission Verification:** Assisting with iterative multi-round code audits, cross-checking numerical consistency between pipeline outputs and report prose, refactoring test suites, and maintaining automated integrity verification scripts (`verify_submission.py`).

#### 2. Human Direction, Auditing, and Critical Validation
The human team maintained active direction, editorial control, and rigorous review throughout all stages:

- **Project Direction & Scope:** The human team selected the research question, chosen asset universe (SPY, QQQ, TLT, GLD, HYG), portfolio weighting scheme, and out-of-sample evaluation horizons (1-day, 5-day, 20-day).
- **Rigorous Verification & Auditing:** The human team ran independent local executions, compared fresh runs against report outputs, detected and corrected numerical discrepancies across different library environments, and ensured all automated test suites pass.
- **Econometric Scrutiny & Reality Checks:** The human team critically challenged model claims--specifically insisting on the introduction of the Gaussian copula benchmark to isolate genuine excess tail dependence from background linear correlation, and demanding an honest, unvarnished evaluation of H-TCM performance during calm out-of-sample periods.
- **Removal of Fabricated Artifacts:** The human team ensured the complete removal of any synthetic roles, exaggerated claims, or unsubstantiated marketing language, presenting a truthful and scientifically defensible report.

#### 3. Authorship and Accountability
In strict compliance with competition rules, the human team has thoroughly reviewed and understands every model, formula, algorithm, and figure. We accept full and undivided responsibility for every methodological decision, line of code, statistical calculation, and policy statement contained in this submission.

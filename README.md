# Risk Across Tails and Timescales

**Team Nexora:** Imasha Karunathilaka, Aadhila Anees, Sameera Ekanayaka, Praveen Madawalage, Tharindu Dhanushka

> Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?

The full write-up is [report/report.pdf](report/report.pdf). This page explains how to run the code.

## Short answer

We use a five-asset portfolio (SPY 30%, QQQ 20%, TLT 25%, GLD 15%, HYG 10%) on daily data from April 2007 to June 2026. A MODWT wavelet decomposition gives us the same returns seen at horizons from one day to more than three months.

- **Tail dependence changes with horizon for some pairs, but not portfolio-wide.** Strong evidence for equities and credit (lower-tail co-exceedance rises from 0.53 at one day to 0.77 beyond 64 days, and the 90% bootstrap interval for the change excludes zero across 63-, 126-, and 252-day blocks), suggestive for equities and Treasuries (hedge weakens, but interval includes zero), and not detectable for gold or for the portfolio as a whole.
- **Horizon-matched dependence gives 5% higher VaR.** A copula fitted to the matching horizon would give about 5% higher 20-day VaR on average than a daily copula (and more than 10% higher on one day in ten). This compares two model forecasts on the same dates; out-of-sample evidence that the higher number is better is weak, as the Diebold-Mariano test cannot statistically separate them ($p = 0.32$ at 20 days, and at 5 days the daily sqrt(h) model actually scores best). The usual sqrt(h) rule hides this because it overstates 20-day VaR by even more.
- **Recommendation.** For positions held 20 days or longer, replace sqrt(h) scaling with a full-horizon GARCH volatility forecast and a copula fitted to the matching wavelet horizon view. In a rolling out-of-sample backtest from 2011 to 2026 this achieved the best point-estimate FZ score at 20 days (without statistical significance) and needed about 5% less capital than sqrt(h).

All numbers above are produced by `run_all.py`. The exact values used in the report are written to `tables/key_numbers.tex`.

## Running it

Python 3.12 is required (all pinned dependencies are tested on Python 3.12; Python 3.13 is unsupported).

```bash
git clone https://github.com/Imashaidk/QuantEdge1.0.git
cd QuantEdge1.0
python3.12 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py                # about 6 minutes
pytest tests                     # about 30 seconds
```

`run_all.py` reads the prices stored in `data/`, so no internet connection is needed. It rebuilds everything in `results/`, `figures/` and `tables/`, and gives identical output on repeated runs on one machine (runs across different hardware or BLAS libraries may differ in a final digit due to floating-point rounding).

To rebuild the PDF after a run:

```bash
cd report
pdflatex report.tex
pdflatex report.tex
```

Before submitting, check everything and build the ZIP:

```bash
python verify_submission.py      # files, report numbers, page count, characters
python scripts/make_zip.py       # writes dist/Nexora_risk_across_tails_and_timescales.zip
```

## What `run_all.py` does

1. Loads daily prices and computes log returns.
2. MODWT multiresolution analysis (sym8, 5 levels) and the share of variance at each scale.
3. Lower and upper tail co-exceedance for the risky and hedge sleeves and six asset pairs, at each wavelet horizon view, with moving block bootstrap intervals and a Gaussian copula benchmark. Also a five-family copula comparison by BIC.
4. Rolling out-of-sample backtest of five VaR models at 1, 5 and 20 days (and the capital comparison at 60 days): 1,000-day window, refit every 21 days, daily GARCH updates.
5. VaR and ES by model on the latest window.
6. Figures, LaTeX tables and `tables/key_numbers.tex`.
7. A printed summary with the recommendation.

## Layout

```text
data/          raw_prices.csv and log_returns.csv (April 2007 to June 2026)
src/
  config.py           tickers, weights and every model setting
  data_loader.py      loads the stored prices and returns
  wavelets.py         MODWT and multiresolution analysis
  tail_dependence.py  horizon views, co-exceedance, block bootstrap
  copulas.py          Gaussian, t, Clayton, Gumbel and Frank copulas
  margins.py          EVT tails on GARCH residuals
  horizon_var.py      the VaR models compared in the report
  backtest.py         Kupiec, Christoffersen and FZ loss statistical tests
  rolling_backtest.py rolling forecasts, coverage tests, FZ score, DM test
  key_numbers.py      numbers quoted in the report text
  visualizer.py       figures and LaTeX tables
tests/         pytest suite
results/       CSV output of every step
figures/       figures used in the report
tables/        LaTeX tables and key numbers used in the report
report/        report.tex and report.pdf
docs/          AI_DISCLOSURE.md
scripts/       make_zip.py
verify_submission.py  pre-submission checks
```

## Data

Daily adjusted closing prices from Yahoo Finance for SPY, QQQ, TLT, GLD and HYG. The sample starts on 11 April 2007, the first day HYG traded. If `data/raw_prices.csv` is removed, `src/data_loader.py` downloads it again with `yfinance`.

## Use of AI tools

AI assistants were used for coding help, debugging and drafting. The team made the modelling decisions, checked the results and is responsible for everything here. See [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md).

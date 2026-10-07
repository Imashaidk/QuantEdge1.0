"""Numbers quoted in the report text.

Every figure the report mentions in its prose comes from here, as a LaTeX macro
in tables/key_numbers.tex. Re-running run_all.py rewrites the file, so the text
can never drift away from the results.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Dict

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import pandas as pd

from src.config import (
    REFIT_EVERY,
    ROLLING_WINDOW,
    TABLES_DIR,
    TAIL_BOOTSTRAP_REPS,
    TAIL_QUANTILE,
    VAR_SIMULATIONS,
)
from src.tail_dependence import BOOTSTRAP_BLOCK


def _pct(x: float, signed: bool = True) -> str:
    return f"{x * 100:+.1f}\\%" if signed else f"{x * 100:.1f}\\%"


def _num(x: float, digits: int = 2, signed: bool = False) -> str:
    x = round(float(x), digits) + 0.0  # avoids printing -0.00
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def collect_key_numbers(
    returns: pd.DataFrame,
    tail: Dict[str, pd.DataFrame],
    var_share: pd.DataFrame,
    forecasts: pd.DataFrame,
    evaluation: pd.DataFrame,
    dm: pd.DataFrame,
    gap: pd.DataFrame,
    stress: pd.DataFrame,
) -> Dict[str, str]:
    """Builds the macro name -> text mapping. Macro names must be letters only."""
    k: Dict[str, str] = {}

    # Settings, so the method section always matches the code
    k["RollingWindow"] = f"{ROLLING_WINDOW:,}"
    k["RefitEvery"] = str(REFIT_EVERY)
    k["BootReps"] = str(TAIL_BOOTSTRAP_REPS)
    k["BootBlock"] = str(BOOTSTRAP_BLOCK)
    k["TailLevel"] = _pct(TAIL_QUANTILE, signed=False).replace(".0", "")
    k["Sims"] = f"{VAR_SIMULATIONS:,}"

    k["SampleStart"] = returns.index[0].strftime("%-d %B %Y")
    k["SampleEnd"] = returns.index[-1].strftime("%-d %B %Y")
    k["SampleDays"] = f"{len(returns):,}"
    k["ForecastStart"] = forecasts["date"].min().strftime("%B %Y")
    k["ForecastDays"] = f"{forecasts['date'].nunique():,}"

    k["FineShare"] = _pct(var_share.loc[["D1", "D2"]].sum().mean() / 100, signed=False)
    k["CoarseShare"] = _pct(var_share.loc[["D5", "S5"]].sum().mean() / 100, signed=False)

    sl = tail["sleeves"].set_index("view")
    last = int(sl.index.max())
    k["SleeveDaily"] = _num(sl.loc[0, "lambda_L"])
    k["SleeveLong"] = _num(sl.loc[last, "lambda_L"])
    k["SleeveChange"] = _num(sl.loc[last, "change_vs_daily"], signed=True)
    k["SleeveChangeLow"] = _num(sl.loc[last, "change_ci_low"], signed=True)
    k["SleeveChangeHigh"] = _num(sl.loc[last, "change_ci_high"], signed=True)
    k["SleeveGaussDaily"] = _num(sl.loc[0, "gauss"])
    k["SleeveUpperDaily"] = _num(sl.loc[0, "lambda_U"])

    pairs = tail["pairs"].set_index(["pair", "view"])
    for pair, name in [("SPY-TLT", "SpyTlt"), ("SPY-HYG", "SpyHyg"), ("SPY-QQQ", "SpyQqq"), ("SPY-GLD", "SpyGld")]:
        k[f"{name}Daily"] = _num(pairs.loc[(pair, 0), "lambda_L"])
        k[f"{name}Long"] = _num(pairs.loc[(pair, last), "lambda_L"])
        k[f"{name}LongLow"] = _num(pairs.loc[(pair, last), "ci_low"])
        k[f"{name}LongHigh"] = _num(pairs.loc[(pair, last), "ci_high"])
        k[f"{name}Change"] = _num(pairs.loc[(pair, last), "change_vs_daily"], signed=True)
        k[f"{name}ChangeLow"] = _num(pairs.loc[(pair, last), "change_ci_low"], signed=True)
        k[f"{name}ChangeHigh"] = _num(pairs.loc[(pair, last), "change_ci_high"], signed=True)
        k[f"{name}GaussLong"] = _num(pairs.loc[(pair, last), "gauss"])

    cop = tail["copulas"]
    k["TWins"] = str(int((cop["best_copula"] == "student_t").sum()))
    k["CopulaFits"] = str(len(cop))

    g = gap.set_index(["model", "horizon"])
    for h, word in [(5, "Five"), (20, "Twenty"), (60, "Sixty")]:
        # means and the top decile go into sentences, so no plus sign; ranges keep it
        k[f"Gap{word}"] = _pct(g.loc[("horizon_copula", h), "var_ratio_mean"], signed=False)
        k[f"Gap{word}Top"] = _pct(g.loc[("horizon_copula", h), "var_ratio_p90"], signed=False)
        k[f"Gap{word}Low"] = _pct(g.loc[("horizon_copula", h), "var_ratio_p10"])
        k[f"Gap{word}High"] = _pct(g.loc[("horizon_copula", h), "var_ratio_p90"])
        k[f"Sqrt{word}"] = _pct(g.loc[("daily_sqrt", h), "var_ratio_mean"], signed=False)

    ev = evaluation.set_index(["model", "horizon"])
    k["ExpectedOne"] = f"{ev.loc[('daily_copula', 1), 'expected']:.0f}"
    for model, name in [("historical", "Hist"), ("gaussian_sqrt", "Gauss"), ("daily_copula", "Copula")]:
        k[f"BreachOne{name}"] = str(int(ev.loc[(model, 1), "breaches"]))
        k[f"Zone{name}"] = str(ev.loc[(model, 1), "basel_zone"]).lower()
    k["KupiecOneCopula"] = _num(ev.loc[("daily_copula", 1), "kupiec_p"])
    k["ChristOneCopula"] = _num(ev.loc[("daily_copula", 1), "christoffersen_p"])

    k["ObsTwenty"] = str(int(ev.loc[("horizon_copula", 20), "obs"]))
    for model, name in [("daily_sqrt", "Sqrt"), ("daily_copula", "Daily"), ("horizon_copula", "Horizon")]:
        k[f"AvgVarTwenty{name}"] = _pct(ev.loc[(model, 20), "avg_var"], signed=False)
        k[f"BreachTwenty{name}"] = str(int(ev.loc[(model, 20), "breaches"]))
    vs_sqrt = ev.loc[("horizon_copula", 20), "avg_var"] / ev.loc[("daily_sqrt", 20), "avg_var"] - 1.0
    k["HorizonVsSqrtTwenty"] = _pct(vs_sqrt)
    k["HorizonSavingTwenty"] = _pct(abs(vs_sqrt), signed=False)
    k["ExpectedTwenty"] = f"{ev.loc[('horizon_copula', 20), 'expected']:.1f}"

    d = dm.set_index(["base", "model", "horizon"])
    for h, word in [(5, "Five"), (20, "Twenty")]:
        k[f"DmDaily{word}"] = _num(d.loc[("daily_copula", "horizon_copula", h), "p_value"])
        k[f"DmSqrt{word}"] = _num(d.loc[("daily_sqrt", "horizon_copula", h), "p_value"])

    st = stress.set_index(["period", "horizon", "model"])
    for period, name in [("2022 rates shock", "TwentyTwo"), ("COVID 2020", "Covid")]:
        for model, mname in [("daily_sqrt", "Sqrt"), ("daily_copula", "Daily"), ("horizon_copula", "Horizon")]:
            k[f"Stress{name}{mname}"] = str(int(st.loc[(period, 20, model), "breaches"]))
        k[f"Stress{name}Days"] = str(int(st.loc[(period, 20, "daily_copula"), "days"]))

    return k


def write_key_numbers(numbers: Dict[str, str], out_dir: Path = TABLES_DIR) -> Path:
    """Writes the numbers as \\newcommand macros, one per line."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "key_numbers.tex"
    lines = ["% Written by run_all.py. Do not edit by hand."]
    for name, value in numbers.items():
        if not name.isalpha():
            raise ValueError(f"LaTeX macro names must be letters only: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")
    return path

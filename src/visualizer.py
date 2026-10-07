"""Figures and LaTeX tables for the report.

Figures, written to figures/:
  1. MODWT multiresolution analysis of SPY and TLT
  2. Tail co-exceedance by horizon view, sleeves and asset pairs
  3. Realised 20-day losses against the rolling VaR forecasts
  4. VaR change from using the horizon copula instead of the daily copula

Tables, written to tables/: variance by scale, tail co-exceedance by horizon,
and the rolling backtest.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Dict

# Set headless backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, SCALE_HORIZONS, TABLES_DIR


plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E0",
    "axes.linewidth": 0.8,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_fig1_wavelet_mra(
    wavelet_dict: Dict[str, pd.DataFrame],
    df_raw: pd.DataFrame,
    primary_asset: str = "SPY",
    secondary_asset: str = "TLT",
    out_path: Path = FIGURES_DIR / "fig1_wavelet_mra_decomposition.png",
    dpi: int = 300,
) -> None:
    """Figure 1: the MRA of two assets, one panel per scale.

    Each panel has its own y-axis, since the coarse scales are much smaller than
    the daily returns.
    """
    scale_keys = [k for k in ["D1", "D2", "D3", "D4", "D5", "S5"] if k in wavelet_dict]
    fig, axes = plt.subplots(len(scale_keys) + 1, 1, figsize=(10, 8.5), sharex=True)
    fig.patch.set_facecolor("white")
    dates = df_raw.index

    panels = [("Daily returns", df_raw)] + [(f"{k}: {SCALE_HORIZONS.get(k, k)}", wavelet_dict[k]) for k in scale_keys]
    for ax, (title, data) in zip(axes, panels):
        ax.plot(dates, data[primary_asset] * 100, color=SERIES_COLORS[0], linewidth=0.6, label=primary_asset)
        if secondary_asset in data.columns:
            ax.plot(dates, data[secondary_asset] * 100, color=SERIES_COLORS[1], linewidth=0.6, alpha=0.8, label=secondary_asset)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7.5)
        ax.grid(True, axis="y")

    axes[0].legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    fig.supylabel("Return, %", fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 1 -> {out_path}")


FIG2_PAIRS = ["SPY-QQQ", "SPY-HYG", "SPY-TLT", "SPY-GLD", "TLT-HYG", "TLT-GLD"]
FIG2_COLORS = ["#8e44ad", "#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#7f8c8d"]


def plot_fig2_tail_by_horizon(
    sleeve_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "fig2_tail_dependence_vs_horizon.png",
    dpi: int = 300,
) -> None:
    """Figure 2: lower-tail co-exceedance across horizon views.

    Left: risky sleeve vs hedge sleeve with its 90% bootstrap band, the upper tail,
    and what a Gaussian copula with the same correlation would give.
    Right: all six asset pairs evaluated across the 30 comparisons.
    """
    sl = sleeve_table.sort_values("view")
    x = sl["view"].to_numpy()
    labels = sl["horizon"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.patch.set_facecolor("white")

    ax1.fill_between(x, sl["ci_low"], sl["ci_high"], color=SERIES_COLORS[0], alpha=0.15, linewidth=0,
                     label="90% block bootstrap band")
    ax1.plot(x, sl["lambda_L"], color=SERIES_COLORS[0], linewidth=2, marker="o", markersize=5,
             label="Lower tail (joint losses)")
    ax1.plot(x, sl["lambda_U"], color=SERIES_COLORS[1], linewidth=2, marker="s", markersize=5,
             label="Upper tail (joint gains)")
    ax1.plot(x, sl["gauss"], color="#7a7a75", linewidth=1.5, linestyle="--",
             label="Gaussian copula, same correlation")
    ax1.axhline(0.05, color="#b5b4ac", linewidth=0.8, linestyle=":")
    ax1.text(x[-1], 0.05, "independence", fontsize=7, color="#7a7a75", ha="right", va="bottom")
    ax1.set_title("Risky sleeve (SPY, QQQ, HYG) vs hedge sleeve (TLT, GLD)", fontsize=9.5, loc="left")
    ax1.set_ylabel("Tail co-exceedance at 5%", fontsize=9)
    ax1.set_ylim(0, max(0.5, float(sl["ci_high"].max()) * 1.1))

    for k, pair in enumerate(FIG2_PAIRS):
        pt = pair_table[pair_table["pair"] == pair].sort_values("view")
        if pt.empty:
            continue
        ax2.plot(pt["view"], pt["lambda_L"], color=FIG2_COLORS[k], linewidth=1.8, marker="o", markersize=4.5, label=pair)
    ax2.axhline(0.05, color="#b5b4ac", linewidth=0.8, linestyle=":")
    ax2.set_title("Lower tail, all asset pairs (6 pairs, 30 tests)", fontsize=9.5, loc="left")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-0.3, x[-1] + 0.3)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Horizon view (moves lasting longer than)", fontsize=8.5)
        ax.grid(True, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    ax1.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 2 -> {out_path}")


def _signed(x: float) -> str:
    """Two decimals with a sign. Rounds first so -0.004 prints as +0.00, not -0.00."""
    return f"{round(float(x), 2) + 0.0:+.2f}"


def export_tail_table(
    sleeve_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    out_dir: Path = TABLES_DIR,
) -> None:
    """LaTeX table of the horizon results that the report quotes."""
    sl = sleeve_table.sort_values("view").set_index("view")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "Horizon & Risky vs hedge & Change vs daily & Gaussian & SPY-HYG & SPY-TLT & SPY-GLD & SPY-QQQ & TLT-HYG & TLT-GLD \\\\",
        "\\midrule",
    ]
    for j, r in sl.iterrows():
        def _get_lambda(pair_name: str) -> str:
            sub = pair_table[(pair_table["pair"] == pair_name) & (pair_table["view"] == j)]
            return f"{sub['lambda_L'].iloc[0]:.2f}" if not sub.empty else "--"

        change = "--" if j == 0 else f"{_signed(r['change_vs_daily'])} [{_signed(r['change_ci_low'])}, {_signed(r['change_ci_high'])}]"
        lines.append(
            f"{r['horizon']} & {r['lambda_L']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {change} & {r['gauss']:.2f} & "
            f"{_get_lambda('SPY-HYG')} & {_get_lambda('SPY-TLT')} & {_get_lambda('SPY-GLD')} & "
            f"{_get_lambda('SPY-QQQ')} & {_get_lambda('TLT-HYG')} & {_get_lambda('TLT-GLD')} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Lower-tail co-exceedance at the 5\\% level by horizon view. Brackets are 90\\% moving block bootstrap intervals.}",
        "\\label{tab:tail_by_horizon}",
        "\\end{table}",
        "",
    ]
    path = out_dir / "tail_by_horizon.tex"
    path.write_text("\n".join(lines).replace("> ", "$>$ "), encoding="utf-8")
    print(f"[Visualizer] Exported LaTeX Table -> {path}")



MODEL_COLORS = {
    "daily_copula": SERIES_COLORS[1],
    "horizon_copula": SERIES_COLORS[0],
    "daily_sqrt": SERIES_COLORS[2],
    "historical": SERIES_COLORS[3],
    "gaussian_sqrt": "#7a7a75",
}


def plot_fig3_capital_gap(
    gap: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "fig3_capital_gap_by_horizon.png",
    dpi: int = 300,
) -> None:
    """Figure 3: how much VaR changes when the dependence matches the horizon.

    Bars are the average change of the horizon copula VaR against the daily copula
    VaR over all rolling forecasts; whiskers are the 10th and 90th percentiles.
    """
    g = gap[(gap["model"] == "horizon_copula") & (gap["horizon"] > 1)].sort_values("horizon")
    x = np.arange(len(g))
    mean = g["var_ratio_mean"].to_numpy() * 100
    lo = mean - g["var_ratio_p10"].to_numpy() * 100
    hi = g["var_ratio_p90"].to_numpy() * 100 - mean

    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor("white")
    ax.bar(x, mean, width=0.5, color=SERIES_COLORS[0], yerr=[lo, hi], capsize=4,
           error_kw={"elinewidth": 1, "ecolor": "#2d2d2a"})
    for xi, m in zip(x, mean):
        ax.text(xi + 0.28, m, f"{m:+.1f}%", fontsize=8, va="center", color="#2d2d2a")
    ax.axhline(0, color="#2d2d2a", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} days" for h in g["horizon"]], fontsize=8.5)
    ax.set_ylabel("VaR change vs daily copula, %", fontsize=8.5)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 3 -> {out_path}")


def plot_fig4_rolling_var(
    forecasts: pd.DataFrame,
    horizon: int = 20,
    out_path: Path = FIGURES_DIR / "fig4_backtest_var_exceedances.png",
    dpi: int = 300,
) -> None:
    """Figure 4: realised h-day loss against the daily and horizon copula VaR."""
    from src.horizon_var import MODEL_LABELS

    f = forecasts[forecasts["h"] == horizon]
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("white")

    loss = f[f["model"] == "horizon_copula"].set_index("date")["loss"]
    ax.plot(loss.index, loss * 100, color="#9a998f", linewidth=0.7, label=f"Realised {horizon}-day loss")
    for name in ["daily_copula", "horizon_copula"]:
        m = f[f["model"] == name].set_index("date")
        ax.plot(m.index, m["VaR"] * 100, color=MODEL_COLORS[name], linewidth=1.4, label=f"VaR 99%, {MODEL_LABELS[name]}")
        hit = m[m["loss"] > m["VaR"]]
        ax.scatter(hit.index, hit["loss"] * 100, s=14, color=MODEL_COLORS[name], zorder=5,
                   marker="o" if name == "horizon_copula" else "x", linewidths=1.2)

    # A few crisis days push the VaR far up; cap the axis so the rest stays readable.
    top = float(np.percentile(f[f["model"] == "horizon_copula"]["VaR"], 99.5)) * 100 * 1.15
    ax.set_ylim(min(-5.0, float(loss.min()) * 100 * 1.1), top)
    ax.set_ylabel("Portfolio loss, % (axis capped)", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 4 -> {out_path}")


# Aliases for backward compatibility
plot_fig3_rolling_var = plot_fig4_rolling_var
plot_fig4_capital_gap = plot_fig3_capital_gap


def export_rolling_backtest_table(
    evaluation: pd.DataFrame,
    dm: pd.DataFrame,
    out_dir: Path = TABLES_DIR,
) -> None:
    """LaTeX table of the rolling backtest."""
    from src.horizon_var import MODEL_LABELS

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "$h$ & Model & Breaches / exp. & Kupiec $p$ & Christ. $p$ & FZ & $\\Delta$FZ ($p$) & Avg VaR & Basel \\\\",
        "\\midrule",
    ]
    for h, eh in evaluation.groupby("horizon"):
        for _, r in eh.iterrows():
            label = MODEL_LABELS[r["model"]]
            if h == 1 and r["model"] in ("daily_sqrt", "horizon_copula"):
                continue  # at one day the three copula models are the same forecast
            if h == 1 and r["model"] == "daily_copula":
                label = "All three copula models"
            d = dm[(dm["horizon"] == h) & (dm["model"] == r["model"])]
            dfz = "--" if r["model"] == "daily_copula" or d.empty else f"{d['mean_fz_diff'].iloc[0]:+.3f} ({d['p_value'].iloc[0]:.2f})"
            zone = r.get("basel_zone", "")
            zone = zone.capitalize() if isinstance(zone, str) and h == 1 else "--"
            lines.append(
                f"{h}d & {label} & {r['breaches']} / {r['expected']:.1f} & {r['kupiec_p']:.2f} & "
                f"{r['christoffersen_p']:.2f} & {r['fz']:.3f} & {dfz} & {r['avg_var'] * 100:.2f}\\% & {zone} \\\\"
            )
        lines.append("\\midrule" if h != evaluation["horizon"].max() else "\\bottomrule")
    lines += [
        "\\end{tabular}",
        "\\caption{Rolling out-of-sample backtest of 99\\% VaR and ES. Coverage tests use non-overlapping windows. "
        "FZ is the Fissler-Ziegel score (lower is better); $\\Delta$FZ is the difference to the baseline model (Daily copula, $h$-day vol) with its Diebold-Mariano $p$-value.}",
        "\\label{tab:backtest}",
        "\\end{table}",
        "",
    ]
    path = out_dir / "backtest_metrics.tex"
    path.write_text("\n".join(lines).replace("sqrt(h)", "$\\sqrt{h}$"), encoding="utf-8")
    print(f"[Visualizer] Exported LaTeX Table -> {path}")


def export_variance_table(variance_decomp_df: pd.DataFrame, out_dir: Path = TABLES_DIR) -> None:
    """LaTeX table of the share of variance in each wavelet scale."""
    caption = "Share of each asset's return variance in each MODWT scale (\\%)."
    label = "tab:variance_decomposition"
    var_tex = variance_decomp_df.round(1).to_latex(
        position="H",
        float_format="%.1f",
    )
    var_tex = var_tex.replace("\\begin{tabular}", "\\centering\n\\small\n\\begin{tabular}", 1)
    var_tex = var_tex.replace("\\end{tabular}", f"\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}", 1)
    path = out_dir / "variance_decomposition.tex"
    path.write_text(var_tex, encoding="utf-8")
    print(f"[Visualizer] Exported LaTeX Table -> {path}")

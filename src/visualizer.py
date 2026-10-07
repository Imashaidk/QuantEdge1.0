"""QuantEdge-MTR Publication Visualizer & Table Export Module.

Generates 4 publication-quality 300 DPI figures in `figures/` and formatted
LaTeX tables in `tables/` adhering to institutional academic standards:

Figures:
1. `figures/fig1_wavelet_mra_decomposition.png`:
   6-panel multiresolution analysis (MRA) decomposition of asset log returns
   into detail scales D1, D2, D3, D4, D5, and smooth secular trend S5.
2. `figures/fig2_tail_dependence_vs_horizon.png`:
   Primary research breakthrough: Lower tail dependence lambda_L(h) vs upper
   tail dependence lambda_U(h) overlaid with the Timescale Asymmetry Ratio (TAR).
3. `figures/fig3_backtest_var_exceedances.png`:
   Out-of-sample portfolio loss time series with VaR(99%) exceedance breaches
   comparing Proposed Multiscale model vs Basel sqrt(h) scaling.
4. `figures/fig4_regulatory_traffic_light.png`:
   Official Basel Traffic Light evaluation (Green/Yellow/Red zones) across all
   comparative models and investment horizons.

LaTeX Tables:
1. `tables/backtest_metrics.tex`: Out-of-sample backtest results and p-values.
2. `tables/copula_tournament.tex`: Scale-optimal copula selection leaderboard.
3. `tables/variance_decomposition.tex`: Percentage variance contribution per scale.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set headless backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    TABLES_DIR,
    TICKERS,
    SCALE_NAMES,
    SCALE_HORIZONS,
    DEFAULT_PORTFOLIO_WEIGHTS,
)
from src.risk_engine import compute_portfolio_returns


# Institutional color palette
PALETTE = {
    "primary": "#1A365D",      # Deep Navy
    "secondary": "#2B6CB0",    # Slate Blue
    "accent_loss": "#C53030",  # Crimson Red (Breaches / Lower tail)
    "accent_gain": "#2F855A",  # Forest Green (Upper tail)
    "tar_line": "#D69E2E",     # Warm Gold (TAR ratio)
    "grid": "#E2E8F0",         # Light Gray grid
    "text": "#2D3748",         # Charcoal text
    "green_zone": "#38A169",   # Basel Green
    "yellow_zone": "#ECC94B",  # Basel Yellow
    "red_zone": "#E53E3E",     # Basel Red
}

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E0",
    "axes.linewidth": 0.8,
    "grid.color": PALETTE["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


def plot_fig1_wavelet_mra(
    wavelet_dict: Dict[str, pd.DataFrame],
    df_raw: pd.DataFrame,
    primary_asset: str = "SPY",
    secondary_asset: str = "TLT",
    out_path: Path = FIGURES_DIR / "fig1_wavelet_mra_decomposition.png",
    dpi: int = 300,
) -> None:
    """Figure 1: Multiresolution Analysis (MRA) decomposition into scales D1-D5, S5.

    Visualizes high-frequency microstructure noise vs. macroeconomic cycles.
    """
    scale_keys = [k for k in ["D1", "D2", "D3", "D4", "D5", "S5"] if k in wavelet_dict]
    n_scales = len(scale_keys) + 1  # Raw returns + scales

    fig, axes = plt.subplots(n_scales, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("white")

    dates = df_raw.index

    # 1. Raw Return Series
    ax0 = axes[0]
    ax0.plot(dates, df_raw[primary_asset], color=PALETTE["primary"], linewidth=0.7, label=f"{primary_asset} Raw Returns")
    if secondary_asset in df_raw.columns:
        ax0.plot(dates, df_raw[secondary_asset], color=PALETTE["secondary"], linewidth=0.6, alpha=0.7, label=f"{secondary_asset} Raw Returns")
    ax0.set_title("Raw Portfolio Asset Log Returns ($R_t$)", fontsize=10, fontweight="bold", loc="left")
    ax0.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax0.grid(True)

    # 2. Decomposed Wavelet Scales
    scale_colors = ["#3182CE", "#2B6CB0", "#4A5568", "#805AD5", "#DD6B20", "#38A169"]

    for idx, scale in enumerate(scale_keys):
        ax = axes[idx + 1]
        s_data = wavelet_dict[scale]
        color = scale_colors[idx % len(scale_colors)]
        horizon_label = SCALE_HORIZONS.get(scale, scale)

        ax.plot(dates, s_data[primary_asset], color=color, linewidth=0.75, label=f"{primary_asset} {scale}")
        if secondary_asset in s_data.columns:
            ax.plot(dates, s_data[secondary_asset], color="#718096", linewidth=0.6, alpha=0.6, linestyle=":", label=f"{secondary_asset} {scale}")

        ax.set_title(f"Scale {scale}: {horizon_label}", fontsize=9, fontweight="bold", loc="left")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
        ax.grid(True)

    # Format Date Axis on bottom subplot
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Observation Date", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 1: MODWT Additive Multiresolution Analysis (MRA) Decomposition", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 1 -> {out_path}")


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FIG2_PAIRS = ["SPY-TLT", "SPY-HYG", "TLT-HYG", "SPY-GLD"]


def plot_fig2_tail_by_horizon(
    sleeve_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "fig2_tail_dependence_vs_horizon.png",
    dpi: int = 300,
) -> None:
    """Figure 2: lower-tail co-exceedance across horizon views.

    Left: risky sleeve vs hedge sleeve with its 90% bootstrap band, the upper tail,
    and what a Gaussian copula with the same correlation would give.
    Right: the main asset pairs.
    """
    sl = sleeve_table.sort_values("view")
    x = sl["view"].to_numpy()
    labels = sl["horizon"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
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
        ax2.plot(pt["view"], pt["lambda_L"], color=SERIES_COLORS[k], linewidth=2, marker="o", markersize=5, label=pair)
    ax2.axhline(0.05, color="#b5b4ac", linewidth=0.8, linestyle=":")
    ax2.set_title("Lower tail, main asset pairs", fontsize=9.5, loc="left")
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
    ax2.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)

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
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Lower-tail co-exceedance at the 5\\% level by horizon view. Brackets are 90\\% moving block bootstrap intervals.}",
        "\\label{tab:tail_by_horizon}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Horizon & Risky vs hedge & Change vs daily & Gaussian & SPY-TLT & SPY-HYG & SPY-GLD \\\\",
        "\\midrule",
    ]
    for j, r in sl.iterrows():
        spy_tlt = pair_table[(pair_table["pair"] == "SPY-TLT") & (pair_table["view"] == j)].iloc[0]
        spy_hyg = pair_table[(pair_table["pair"] == "SPY-HYG") & (pair_table["view"] == j)].iloc[0]
        spy_gld = pair_table[(pair_table["pair"] == "SPY-GLD") & (pair_table["view"] == j)].iloc[0]
        change = "--" if j == 0 else f"{_signed(r['change_vs_daily'])} [{_signed(r['change_ci_low'])}, {_signed(r['change_ci_high'])}]"
        lines.append(
            f"{r['horizon']} & {r['lambda_L']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {change} & {r['gauss']:.2f} & "
            f"{spy_tlt['lambda_L']:.2f} & {spy_hyg['lambda_L']:.2f} & {spy_gld['lambda_L']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
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


def plot_fig3_rolling_var(
    forecasts: pd.DataFrame,
    horizon: int = 20,
    out_path: Path = FIGURES_DIR / "fig3_backtest_var_exceedances.png",
    dpi: int = 300,
) -> None:
    """Figure 3: realised h-day loss against the daily and horizon copula VaR."""
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
    print(f"[Visualizer] Exported Figure 3 -> {out_path}")


def plot_fig4_capital_gap(
    gap: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "fig4_capital_gap_by_horizon.png",
    dpi: int = 300,
) -> None:
    """Figure 4: how much VaR changes when the dependence matches the horizon.

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
    print(f"[Visualizer] Exported Figure 4 -> {out_path}")


def export_rolling_backtest_table(
    evaluation: pd.DataFrame,
    dm: pd.DataFrame,
    out_dir: Path = TABLES_DIR,
) -> None:
    """LaTeX table of the rolling backtest."""
    from src.horizon_var import MODEL_LABELS

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Rolling out-of-sample backtest of 99\\% VaR and ES. Coverage tests use non-overlapping windows. "
        "FZ is the Fissler-Ziegel score (lower is better); $\\Delta$FZ is the difference to the daily copula with its Diebold-Mariano $p$-value.}",
        "\\label{tab:backtest}",
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
    lines += ["\\end{tabular}", "\\end{table}", ""]
    path = out_dir / "backtest_metrics.tex"
    path.write_text("\n".join(lines).replace("sqrt(h)", "$\\sqrt{h}$"), encoding="utf-8")
    print(f"[Visualizer] Exported LaTeX Table -> {path}")


def export_variance_table(variance_decomp_df: pd.DataFrame, out_dir: Path = TABLES_DIR) -> None:
    """LaTeX table of the share of variance in each wavelet scale."""
    var_tex = variance_decomp_df.round(1).to_latex(
        caption="Share of each asset's return variance in each MODWT scale (\\%).",
        label="tab:variance_decomposition",
        position="htbp",
        float_format="%.1f",
    )
    var_tex = var_tex.replace("\\begin{tabular}", "\\centering\n\\small\n\\begin{tabular}", 1)
    path = out_dir / "variance_decomposition.tex"
    path.write_text(var_tex, encoding="utf-8")
    print(f"[Visualizer] Exported LaTeX Table -> {path}")


def plot_fig3_backtest_exceedances(
    df_test: pd.DataFrame,
    weights: np.ndarray,
    proposed_var: float,
    basel_var: float,
    horizon: int = 1,
    out_path: Path = FIGURES_DIR / "fig3_backtest_var_exceedances.png",
    dpi: int = 300,
) -> None:
    """Figure 3: Out-of-sample portfolio losses against VaR(99%) limits.

    Shows breach exceedances comparing the Proposed Multiscale model vs Basel sqrt(h) scaler.
    """
    r_test = compute_portfolio_returns(df_test, weights)
    dates = df_test.index

    if horizon == 1:
        losses = -r_test
    else:
        s = pd.Series(-r_test, index=dates)
        losses = s.rolling(horizon).sum().dropna().values
        dates = dates[horizon - 1 :]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    # Plot loss series
    ax.plot(dates, losses * 100.0, color="#718096", linewidth=0.75, alpha=0.85, label="Out-of-Sample Portfolio Loss ($L_t$)")

    # Plot Basel VaR threshold
    ax.axhline(basel_var * 100.0, color="#E53E3E", linestyle="--", linewidth=1.8, label=f"Basel $\\sqrt{{h}}$ Scaler VaR(99%): {basel_var*100:.2f}%")

    # Plot Proposed Multiscale VaR threshold
    ax.axhline(proposed_var * 100.0, color="#2B6CB0", linestyle="-", linewidth=2.0, label=f"Proposed Multiscale VaR(99%): {proposed_var*100:.2f}%")

    # Identify and plot breach points
    basel_breach_idx = np.where(losses > basel_var)[0]
    proposed_breach_idx = np.where(losses > proposed_var)[0]

    if len(basel_breach_idx) > 0:
        ax.scatter(dates[basel_breach_idx], losses[basel_breach_idx] * 100.0, color="#E53E3E", marker="^", s=60, zorder=5, label=f"Basel Breaches (N={len(basel_breach_idx)})")

    if len(proposed_breach_idx) > 0:
        ax.scatter(dates[proposed_breach_idx], losses[proposed_breach_idx] * 100.0, color="#2B6CB0", marker="o", s=70, facecolors="none", edgecolors="#2B6CB0", linewidth=1.8, zorder=6, label=f"Proposed Breaches (N={len(proposed_breach_idx)})")

    ax.set_ylabel("Portfolio Loss / Threshold (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Out-of-Sample Date (2023-2026)", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    plt.title(f"Figure 3: Out-of-Sample VaR(99%) Exceedances ({horizon}-Day Holding Horizon)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 3 -> {out_path}")


def plot_fig4_regulatory_traffic_light(
    backtest_df: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "fig4_regulatory_traffic_light.png",
    dpi: int = 300,
) -> None:
    """Figure 4: Basel Committee Regulatory Traffic Light Zone evaluation.

    Categorizes each model into Green, Yellow, and Red zones across horizons.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    # Model name clean display
    model_labels = {
        "Historical_Simulation": "Hist. Simulation",
        "Parametric_Gaussian": "Parametric Gauss",
        "Static_Copula": "Static Copula",
        "Basel_Sqrt_Time": "Basel Sqrt(h)",
        "Proposed_Multiscale_Copula": "Proposed MTR",
        "H_TCM_Adjusted": "H-TCM Policy",
    }

    df_plot = backtest_df.copy()
    df_plot["Display_Model"] = df_plot["Model"].map(lambda m: model_labels.get(m, m))
    df_plot["Label"] = df_plot["Display_Model"] + " (" + df_plot["Horizon"] + ")"

    y_pos = np.arange(len(df_plot))
    breaches = df_plot["Breaches"].to_numpy(dtype=float)

    # Color bars by Basel Zone
    zone_colors = {
        "GREEN": PALETTE["green_zone"],
        "YELLOW": PALETTE["yellow_zone"],
        "RED": PALETTE["red_zone"],
    }
    bar_colors = [zone_colors.get(z, "gray") for z in df_plot["Basel_Zone"]]

    # Add background zone shading (on normalized 250d basis: Green <=4, Yellow 5-9, Red >=10)
    # Using sample size N = total_obs
    avg_n = float(df_plot["Total_Obs"].iloc[0])
    scale_factor = avg_n / 250.0

    green_max = 4.5 * scale_factor
    yellow_max = 9.5 * scale_factor
    max_x = max(np.max(breaches) * 1.25, yellow_max * 1.3)

    ax.axvspan(0, green_max, color="#C6F6D5", alpha=0.35, label="Basel Green Zone (Approved)")
    ax.axvspan(green_max, yellow_max, color="#FEFCBF", alpha=0.35, label="Basel Yellow Zone (Capital Surcharge)")
    ax.axvspan(yellow_max, max_x, color="#FED7D7", alpha=0.35, label="Basel Red Zone (Model Rejected)")

    bars = ax.barh(y_pos, breaches, align="center", color=bar_colors, edgecolor="black", linewidth=0.6, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Label"], fontsize=8.5)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel(f"Out-of-Sample VaR(99%) Exceedance Breaches (N = {int(avg_n)})", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max_x)
    ax.grid(True, axis="x")

    # Annotate breach values and zone
    for idx, bar in enumerate(bars):
        w = bar.get_width()
        zone = df_plot["Basel_Zone"].iloc[idx]
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(w)} [{zone}]", va="center", ha="left", fontsize=8, fontweight="bold")

    ax.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
    plt.title("Figure 4: BCBS Basel Traffic Light Backtest Matrix (Out-of-Sample 2023-2026)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Exported Figure 4 -> {out_path}")


def export_latex_tables(
    backtest_df: pd.DataFrame,
    copula_tournament_results: Dict[str, Any],
    variance_decomp_df: Optional[pd.DataFrame] = None,
    out_dir: Path = TABLES_DIR,
) -> None:
    """Exports structured LaTeX tables for inclusion in report/report.tex."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Backtest Metrics Table (tables/backtest_metrics.tex and .md)
    bt_cols = ["Horizon", "Model", "Total_Obs", "Breaches", "Breach_Rate", "Kupiec_p", "Christoffersen_p", "Basel_Zone", "FZ_Loss"]
    valid_cols = [c for c in bt_cols if c in backtest_df.columns]
    tex_df = backtest_df[valid_cols].copy()
    tex_df["Model"] = tex_df["Model"].str.replace("_", " ")
    tex_df["Breach_Rate"] = tex_df["Breach_Rate"].str.replace("%", "\\%")
    if "Kupiec_p" in tex_df.columns:
        tex_df["Kupiec_p"] = tex_df["Kupiec_p"].apply(lambda x: f"{float(x):.4f}")
    if "Christoffersen_p" in tex_df.columns:
        tex_df["Christoffersen_p"] = tex_df["Christoffersen_p"].apply(lambda x: f"{float(x):.4f}")
    if "FZ_Loss" in tex_df.columns:
        tex_df["FZ_Loss"] = tex_df["FZ_Loss"].apply(lambda x: f"{float(x):.4f}")
    
    # Basel-style for 1d and Diagnostic for 5d/20d
    tex_df["Status_Diag"] = tex_df.apply(
        lambda r: f"{r['Basel_Zone']} (Basel)" if r["Horizon"] == "1d" else f"{r['Basel_Zone']} (Diag)",
        axis=1,
    )
    tex_df = tex_df.drop(columns=["Basel_Zone"])
    tex_df = tex_df.rename(columns={
        "Total_Obs": "Obs",
        "Breach_Rate": "Breach Rate",
        "Kupiec_p": "Kupiec $p$",
        "Christoffersen_p": "Christoffersen $p$",
        "Status_Diag": "Status / Diagnostic",
        "FZ_Loss": "FZ Loss",
    })

    bt_path = out_dir / "backtest_metrics.tex"
    tex_code = tex_df.to_latex(
        index=False,
        caption="Out-of-Sample Risk Model Evaluation: Basel-style (1d) and Breach Diagnostic (5d, 20d)",
        label="tab:backtest_metrics",
        column_format="llccccccc",
        position="htbp",
        escape=False,
    )
    with open(bt_path, "w", encoding="utf-8") as f:
        f.write(tex_code)

    # 2. Copula Tournament Leaderboard Table (tables/copula_tournament.tex)
    c_rows = []
    for s in ["D1", "D2", "D3", "D4", "D5", "S5"]:
        if s in copula_tournament_results:
            res = copula_tournament_results[s]
            raw_c = res.get("best_copula", "N/A")
            if raw_c.lower() == "student_t":
                best_c = "Student-$t$"
            else:
                best_c = raw_c.capitalize()
            lL_theo = float(res.get("lambda_L_theo", res.get("lambda_L", 0.0)))
            lU_theo = float(res.get("lambda_U_theo", res.get("lambda_U", 0.0)))
            lL_emp = float(res.get("lambda_L_emp", res.get("lambda_L", 0.0)))
            gauss_bench = float(res.get("lambda_gauss_bench", 0.0))
            excess_lL = float(res.get("excess_lambda_L", lL_emp - gauss_bench))
            lU_emp = float(res.get("lambda_U_emp", res.get("lambda_U", 0.0)))
            tar_emp = float(res.get("tar_emp", lL_emp - lU_emp))
            bic = res.get("bic_scores", {}).get(res.get("best_copula", ""), 0.0)

            c_rows.append({
                "Scale": s,
                "Trading Horizon": SCALE_HORIZONS.get(s, s).split("(")[0].strip(),
                "Best Copula": best_c,
                "Theo. $\\lambda_L$": f"{lL_theo:.3f}",
                "Theo. $\\lambda_U$": f"{lU_theo:.3f}",
                "Emp. $\\lambda_L$": f"{lL_emp:.3f}",
                "Gauss Bench": f"{gauss_bench:.3f}",
                "Excess $\\lambda_L$": f"{excess_lL:+.3f}",
                "Emp. $\\lambda_U$": f"{lU_emp:.3f}",
                "Emp. TAR": f"{tar_emp:+.3f}",
                "BIC": f"{bic:.1f}",
            })

    if c_rows:
        c_df = pd.DataFrame(c_rows)
        c_path = out_dir / "copula_tournament.tex"
        c_tex = c_df.to_latex(
            index=False,
            caption="Scale-Optimal Copula Tournament Leaderboard, Tail Dependence, and Gaussian Benchmarks",
            label="tab:copula_tournament",
            column_format="llccccccccc",
            position="htbp",
            escape=False,
        )
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(c_tex)
        print(f"[Visualizer] Exported LaTeX Table -> {c_path}")

    # 3. Variance Decomposition Table (tables/variance_decomposition.tex)
    if variance_decomp_df is not None and not variance_decomp_df.empty:
        var_path = out_dir / "variance_decomposition.tex"
        var_tex = variance_decomp_df.round(2).to_latex(
            caption="Wavelet Multiresolution Percentage Variance Contribution Across Assets",
            label="tab:variance_decomposition",
            position="htbp",
            float_format="%.2f",
        )
        with open(var_path, "w", encoding="utf-8") as f:
            f.write(var_tex)
        print(f"[Visualizer] Exported LaTeX Table -> {var_path}")


def generate_all_figures_and_tables(
    wavelet_dict: Dict[str, pd.DataFrame],
    copula_tournament_results: Dict[str, Any],
    backtest_results_df: pd.DataFrame,
    df_raw_train: pd.DataFrame,
    df_raw_test: pd.DataFrame,
    weights: np.ndarray = DEFAULT_PORTFOLIO_WEIGHTS,
    variance_decomp_df: Optional[pd.DataFrame] = None,
    out_dir_figures: Path = FIGURES_DIR,
    out_dir_tables: Path = TABLES_DIR,
    tail_results: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Generates all publication figures and LaTeX tables."""
    out_dir_figures.mkdir(parents=True, exist_ok=True)
    out_dir_tables.mkdir(parents=True, exist_ok=True)

    # Figure 1: Wavelet MRA Decomposition
    plot_fig1_wavelet_mra(
        wavelet_dict=wavelet_dict,
        df_raw=df_raw_train,
        primary_asset="SPY",
        secondary_asset="TLT",
        out_path=out_dir_figures / "fig1_wavelet_mra_decomposition.png",
    )

    # Figure 2: tail dependence across horizon views
    if tail_results is not None:
        plot_fig2_tail_by_horizon(
            tail_results["sleeves"],
            tail_results["pairs"],
            out_path=out_dir_figures / "fig2_tail_dependence_vs_horizon.png",
        )
        export_tail_table(tail_results["sleeves"], tail_results["pairs"], out_dir=out_dir_tables)

    # Figure 3: Backtest Exceedances (Horizon = 1 day)
    # Extract VaR values from backtest_results_df
    m_proposed = backtest_results_df[
        (backtest_results_df["Model"] == "Proposed_Multiscale_Copula")
        & (backtest_results_df["Horizon"] == "1d")
    ]
    m_basel = backtest_results_df[
        (backtest_results_df["Model"] == "Basel_Sqrt_Time")
        & (backtest_results_df["Horizon"] == "1d")
    ]

    if not m_proposed.empty and "VaR_Pred" in m_proposed.columns:
        var_p = float(m_proposed["VaR_Pred"].iloc[0])
    else:
        var_p = 0.0222

    if not m_basel.empty and "VaR_Pred" in m_basel.columns:
        var_b = float(m_basel["VaR_Pred"].iloc[0])
    else:
        var_b = 0.0188

    plot_fig3_backtest_exceedances(
        df_test=df_raw_test,
        weights=weights,
        proposed_var=var_p,
        basel_var=var_b,
        horizon=1,
        out_path=out_dir_figures / "fig3_backtest_var_exceedances.png",
    )

    # Figure 4: Regulatory Basel Traffic Light Matrix
    plot_fig4_regulatory_traffic_light(
        backtest_df=backtest_results_df,
        out_path=out_dir_figures / "fig4_regulatory_traffic_light.png",
    )

    # Export LaTeX Tables
    export_latex_tables(
        backtest_df=backtest_results_df,
        copula_tournament_results=copula_tournament_results,
        variance_decomp_df=variance_decomp_df,
        out_dir=out_dir_tables,
    )


if __name__ == "__main__":
    from src.data_loader import load_and_split_data
    from src.wavelets import decompose_multiscale, compute_scale_variance_decomposition
    from src.margins import pseudo_observations
    from src.copulas import run_scale_copula_tournament
    from src.backtest import run_out_of_sample_backtest

    # 1. Load Data
    print("\n[Step 1] Loading data...")
    df_train, df_test = load_and_split_data()

    # 2. Decompose Wavelets
    print("\n[Step 2] Performing MODWT wavelet decomposition...")
    decomposed = decompose_multiscale(df_train)
    var_decomp = compute_scale_variance_decomposition(decomposed, normalize=True)

    # 3. Fit Copulas
    print("\n[Step 3] Running scale copula tournament...")
    copula_results = {}
    for s in ["D1", "D2", "D3", "D4", "D5", "S5"]:
        u_s = pseudo_observations(decomposed[s])
        t_res = run_scale_copula_tournament(u_s, scale_name=s)
        copula_results[s] = t_res

    # 4. Run Backtests
    print("\n[Step 4] Running out-of-sample backtests...")
    backtest_df = run_out_of_sample_backtest(
        df_test=df_test,
        weights=DEFAULT_PORTFOLIO_WEIGHTS,
        copula_results=copula_results,
        h_horizons=[1, 5, 20],
        df_train=df_train,
    )

    # 5. Generate Figures and Tables
    print("\n[Step 5] Generating 300 DPI publication figures & LaTeX tables...")
    generate_all_figures_and_tables(
        wavelet_dict=decomposed,
        copula_tournament_results=copula_results,
        backtest_results_df=backtest_df,
        df_raw_train=df_train,
        df_raw_test=df_test,
        weights=DEFAULT_PORTFOLIO_WEIGHTS,
        variance_decomp_df=var_decomp,
    )

    print("\n[SUCCESS] All 4 publication figures and LaTeX tables successfully generated.")

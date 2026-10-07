"""Unit and Artifact Integration Tests for Visualizer Module.

Tests:
1. Generation of Figure 1 (Wavelet MRA Decomposition).
2. Generation of Figure 2 (Tail Dependence vs Horizon & TAR curve).
3. Generation of Figure 3 (Out-of-sample VaR Exceedance plot).
4. Generation of Figure 4 (Basel Regulatory Traffic Light Matrix).
5. Generation of LaTeX Tables (backtest_metrics.tex, copula_tournament.tex).
6. End-to-end execution: generate_all_figures_and_tables execution.

Author: Sameera Ekanayaka
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_PORTFOLIO_WEIGHTS,
    FIGURES_DIR,
    TABLES_DIR,
)
from src.visualizer import (
    export_latex_tables,
    generate_all_figures_and_tables,
    plot_fig1_wavelet_mra,
    plot_fig2_tail_dependence_vs_horizon,
    plot_fig3_backtest_exceedances,
    plot_fig4_regulatory_traffic_light,
)


@pytest.fixture
def mock_visualizer_data(tmp_path):
    """Provides synthetic data for testing plot and table generation."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["SPY", "QQQ", "TLT", "GLD", "HYG"]

    rng = np.random.default_rng(42)
    df_raw = pd.DataFrame(rng.normal(0, 0.01, size=(100, 5)), index=dates, columns=tickers)

    wavelet_dict = {
        s: pd.DataFrame(rng.normal(0, 0.005, size=(100, 5)), index=dates, columns=tickers)
        for s in ["D1", "D2", "D3", "D4", "D5", "S5"]
    }

    copula_results = {
        "D1": {"best_copula": "student_t", "lambda_L": 0.042, "lambda_U": 0.042, "tar": 0.0, "bic_scores": {"student_t": -150.0}},
        "D2": {"best_copula": "student_t", "lambda_L": 0.050, "lambda_U": 0.045, "tar": 0.005, "bic_scores": {"student_t": -140.0}},
        "D3": {"best_copula": "clayton", "lambda_L": 0.120, "lambda_U": 0.010, "tar": 0.110, "bic_scores": {"clayton": -130.0}},
        "D4": {"best_copula": "clayton", "lambda_L": 0.250, "lambda_U": 0.020, "tar": 0.230, "bic_scores": {"clayton": -120.0}},
        "D5": {"best_copula": "gumbel", "lambda_L": 0.560, "lambda_U": 0.040, "tar": 0.520, "bic_scores": {"gumbel": -110.0}},
        "S5": {"best_copula": "gumbel", "lambda_L": 0.810, "lambda_U": 0.020, "tar": 0.790, "bic_scores": {"gumbel": -100.0}},
    }

    backtest_df = pd.DataFrame([
        {"Horizon": "1d", "Model": "Historical_Simulation", "Breaches": 6, "Breach_Rate": "0.69%", "Kupiec_p": 0.32, "Christoffersen_p": 0.77, "Basel_Zone": "GREEN", "FZ_Loss": -0.85, "Total_Obs": 875},
        {"Horizon": "1d", "Model": "Basel_Sqrt_Time", "Breaches": 6, "Breach_Rate": "0.69%", "Kupiec_p": 0.32, "Christoffersen_p": 0.77, "Basel_Zone": "GREEN", "FZ_Loss": -0.85, "Total_Obs": 875},
        {"Horizon": "1d", "Model": "Proposed_Multiscale_Copula", "Breaches": 3, "Breach_Rate": "0.34%", "Kupiec_p": 0.08, "Christoffersen_p": 0.95, "Basel_Zone": "GREEN", "FZ_Loss": -0.92, "Total_Obs": 875},
        {"Horizon": "5d", "Model": "Basel_Sqrt_Time", "Breaches": 12, "Breach_Rate": "1.38%", "Kupiec_p": 0.001, "Christoffersen_p": 0.02, "Basel_Zone": "RED", "FZ_Loss": -0.71, "Total_Obs": 871},
        {"Horizon": "5d", "Model": "Proposed_Multiscale_Copula", "Breaches": 4, "Breach_Rate": "0.46%", "Kupiec_p": 0.15, "Christoffersen_p": 0.88, "Basel_Zone": "GREEN", "FZ_Loss": -0.89, "Total_Obs": 871},
    ])

    return df_raw, wavelet_dict, copula_results, backtest_df, tmp_path


def test_plot_fig1_generation(mock_visualizer_data):
    """Verifies Figure 1 file export."""
    df_raw, wavelet_dict, _, _, tmp_path = mock_visualizer_data
    out_file = tmp_path / "test_fig1.png"

    plot_fig1_wavelet_mra(wavelet_dict, df_raw, out_path=out_file, dpi=100)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_fig2_generation(mock_visualizer_data):
    """Verifies Figure 2 file export."""
    _, _, copula_results, _, tmp_path = mock_visualizer_data
    out_file = tmp_path / "test_fig2.png"

    plot_fig2_tail_dependence_vs_horizon(copula_results, out_path=out_file, dpi=100)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_fig3_generation(mock_visualizer_data):
    """Verifies Figure 3 file export."""
    df_raw, _, _, _, tmp_path = mock_visualizer_data
    out_file = tmp_path / "test_fig3.png"

    plot_fig3_backtest_exceedances(
        df_test=df_raw,
        weights=DEFAULT_PORTFOLIO_WEIGHTS,
        proposed_var=0.015,
        basel_var=0.018,
        horizon=1,
        out_path=out_file,
        dpi=100,
    )
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_plot_fig4_generation(mock_visualizer_data):
    """Verifies Figure 4 file export."""
    _, _, _, backtest_df, tmp_path = mock_visualizer_data
    out_file = tmp_path / "test_fig4.png"

    plot_fig4_regulatory_traffic_light(backtest_df, out_path=out_file, dpi=100)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_export_latex_tables(mock_visualizer_data):
    """Verifies LaTeX table generation."""
    _, _, copula_results, backtest_df, tmp_path = mock_visualizer_data

    export_latex_tables(
        backtest_df=backtest_df,
        copula_tournament_results=copula_results,
        out_dir=tmp_path,
    )

    bt_tex = tmp_path / "backtest_metrics.tex"
    c_tex = tmp_path / "copula_tournament.tex"

    assert bt_tex.exists()
    assert "\\begin{table}" in bt_tex.read_text(encoding="utf-8")

    assert c_tex.exists()
    assert "\\begin{table}" in c_tex.read_text(encoding="utf-8")


def test_generate_all_figures_and_tables(mock_visualizer_data):
    """Verifies generating all figures and tables end-to-end."""
    df_raw, wavelet_dict, copula_results, backtest_df, tmp_path = mock_visualizer_data
    fig_dir = tmp_path / "figures"
    tab_dir = tmp_path / "tables"

    generate_all_figures_and_tables(
        wavelet_dict=wavelet_dict,
        copula_tournament_results=copula_results,
        backtest_results_df=backtest_df,
        df_raw_train=df_raw,
        df_raw_test=df_raw,
        weights=DEFAULT_PORTFOLIO_WEIGHTS,
        out_dir_figures=fig_dir,
        out_dir_tables=tab_dir,
    )

    assert (fig_dir / "fig1_wavelet_mra_decomposition.png").exists()
    assert (fig_dir / "fig2_tail_dependence_vs_horizon.png").exists()
    assert (fig_dir / "fig3_backtest_var_exceedances.png").exists()
    assert (fig_dir / "fig4_regulatory_traffic_light.png").exists()
    assert (tab_dir / "backtest_metrics.tex").exists()
    assert (tab_dir / "copula_tournament.tex").exists()

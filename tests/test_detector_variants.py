from __future__ import annotations

import numpy as np
import pandas as pd

from grb_refresh import BenchmarkConfig, SimulationParams
from run_detector_variants import build_best_variant_selector, build_variant_comparison, run_detector_variants


def _small_variant_scenarios() -> dict[str, SimulationParams]:
    return {
        "mid": SimulationParams(
            A1=800.0,
            tau1=12.0,
            tau_r=1.5,
            B=0.3,
            f_qpo=0.41,
            phi1=0.0,
            N=50,
            Ai=3.0,
            R_bg=25.0,
            t0=0.0,
            k=0.0,
            T=12.0,
            dt=0.02,
            seed=3001,
        ),
        "boat_drift": SimulationParams(
            A1=1200.0,
            tau1=20.0,
            tau_r=2.5,
            B=0.3,
            f_qpo=0.41,
            phi1=0.3,
            N=90,
            Ai=2.0,
            R_bg=30.0,
            t0=0.0,
            k=0.00001,
            T=20.0,
            dt=0.03,
            seed=3002,
            Gamma=True,
        ),
    }


def test_build_variant_comparison_schema_and_baseline_deltas():
    summary_global = pd.DataFrame(
        [
            {"scenario": "mid", "B": 0.0, "p0_scale": 1.0, "recovery_rate_sig": 0.05, "false_positive_rate_sig": 0.05},
            {"scenario": "mid", "B": 0.2, "p0_scale": 1.0, "recovery_rate_sig": 0.30, "false_positive_rate_sig": np.nan},
            {"scenario": "mid", "B": 0.3, "p0_scale": 1.0, "recovery_rate_sig": 0.40, "false_positive_rate_sig": np.nan},
        ]
    )
    summary_windowed = pd.DataFrame(
        [
            {"scenario": "mid", "B": 0.0, "p0_scale": 1.0, "recovery_rate_sig": 0.03, "false_positive_rate_sig": 0.03},
            {"scenario": "mid", "B": 0.2, "p0_scale": 1.0, "recovery_rate_sig": 0.45, "false_positive_rate_sig": np.nan},
            {"scenario": "mid", "B": 0.3, "p0_scale": 1.0, "recovery_rate_sig": 0.50, "false_positive_rate_sig": np.nan},
        ]
    )
    summary_detrended = pd.DataFrame(
        [
            {"scenario": "mid", "B": 0.0, "p0_scale": 1.0, "recovery_rate_sig": 0.02, "false_positive_rate_sig": 0.02},
            {"scenario": "mid", "B": 0.2, "p0_scale": 1.0, "recovery_rate_sig": 0.55, "false_positive_rate_sig": np.nan},
            {"scenario": "mid", "B": 0.3, "p0_scale": 1.0, "recovery_rate_sig": 0.60, "false_positive_rate_sig": np.nan},
        ]
    )
    summary_map = {
        "global_tapered_fft_sig": summary_global,
        "windowed_fft_sig": summary_windowed,
        "detrended_fft_sig": summary_detrended,
    }

    comparison = build_variant_comparison(summary_map)
    expected_columns = {
        "scenario",
        "detector_variant",
        "B_bucket",
        "TPR_sig",
        "FPR_sig",
        "delta_tpr_vs_baseline",
        "delta_fpr_vs_baseline",
        "passes_balanced_bar",
    }
    assert expected_columns.issubset(set(comparison.columns))
    assert set(comparison["B_bucket"].unique().tolist()) == {"B0", "B>=0.2"}

    baseline_rows = comparison[comparison["detector_variant"] == "global_tapered_fft_sig"]
    baseline_bgt = baseline_rows[baseline_rows["B_bucket"] == "B>=0.2"].iloc[0]
    assert np.isclose(float(baseline_bgt["delta_tpr_vs_baseline"]), 0.0, atol=1e-12)
    assert np.isclose(float(baseline_bgt["delta_fpr_vs_baseline"]), 0.0, atol=1e-12)

    selector = build_best_variant_selector(comparison)
    selector_columns = {
        "selector_scope",
        "selector_key",
        "best_detector_variant",
        "TPR_sig",
        "FPR_sig",
        "score_tpr_minus_fpr",
        "passes_balanced_bar",
    }
    assert selector_columns.issubset(set(selector.columns))
    scenario_mid = selector[
        (selector["selector_scope"] == "scenario") & (selector["selector_key"] == "mid")
    ].iloc[0]
    assert scenario_mid["best_detector_variant"] == "detrended_fft_sig"


def test_run_detector_variants_smoke_outputs(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("mid", "boat_drift"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=2,
        n_surrogates=20,
        seed_start=410000,
        alpha=0.05,
        detector_variant="windowed_fft_sig",
        detrend_order=1,
    )

    out_dir = tmp_path / "outputs" / "detector_variants"
    fig_dir = tmp_path / "figures"
    paper_path = tmp_path / "paper" / "grb_substructure_v2.md"
    decision_log_path = tmp_path / "docs" / "sprint3_decision_log.md"

    comparison_df, summary_by_variant = run_detector_variants(
        config=cfg,
        variants=("global_tapered_fft_sig", "windowed_fft_sig", "detrended_fft_sig"),
        output_dir=out_dir,
        figures_dir=fig_dir,
        paper_path=paper_path,
        decision_log_path=decision_log_path,
        scenario_map=_small_variant_scenarios(),
        max_points_for_bb=300,
        max_points_for_sig=300,
        update_paper=True,
    )

    for variant in ("global_tapered_fft_sig", "windowed_fft_sig", "detrended_fft_sig"):
        assert (out_dir / f"run_level_{variant}.csv").exists()
        assert (out_dir / f"summary_{variant}.csv").exists()
        assert variant in summary_by_variant

    assert (out_dir / "variant_comparison.csv").exists()
    assert (out_dir / "best_variant_selector.csv").exists()
    assert (fig_dir / "variant_tpr_fpr_tradeoff.png").exists()
    assert (fig_dir / "variant_roc_like_grid.png").exists()
    assert decision_log_path.exists()
    assert "Sprint outcome:" in decision_log_path.read_text(encoding="utf-8")

    paper_text = paper_path.read_text(encoding="utf-8")
    assert "Detector Variant Comparison" in paper_text
    assert "Best-Variant Selector" in paper_text
    assert "<!-- DETECTOR_VARIANT_SECTION_START -->" in paper_text
    assert "<!-- DETECTOR_VARIANT_SECTION_END -->" in paper_text

    assert not comparison_df.empty
    assert set(comparison_df["B_bucket"].unique().tolist()) == {"B0", "B>=0.2"}

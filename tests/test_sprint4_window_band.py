from __future__ import annotations

import numpy as np

from grb_refresh import BenchmarkConfig, SimulationParams
from run_sprint4_window_band import run_sprint4_window_band


def _small_scenarios() -> dict[str, SimulationParams]:
    return {
        "mid": SimulationParams(
            A1=600.0,
            tau1=10.0,
            tau_r=1.5,
            B=0.3,
            f_qpo=0.41,
            phi1=0.0,
            N=30,
            Ai=2.0,
            R_bg=20.0,
            t0=0.0,
            k=0.0,
            T=10.0,
            dt=0.02,
            seed=5101,
        ),
        "boat_drift": SimulationParams(
            A1=900.0,
            tau1=18.0,
            tau_r=2.0,
            B=0.3,
            f_qpo=0.41,
            phi1=0.1,
            N=50,
            Ai=2.0,
            R_bg=25.0,
            t0=0.0,
            k=0.00001,
            T=16.0,
            dt=0.03,
            seed=5102,
            Gamma=True,
        ),
    }


def test_run_sprint4_window_band_smoke(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("mid", "boat_drift"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=2,
        n_surrogates=20,
        alpha=0.05,
        seed_start=600000,
    )

    out_dir = tmp_path / "outputs" / "sprint4_window_band"
    fig_dir = tmp_path / "figures"
    log_path = tmp_path / "docs" / "sprint4_window_band_log.md"

    candidate_df, best_df = run_sprint4_window_band(
        config=cfg,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        band_grid=[(0.35, 0.45), (0.30, 0.40)],
        window_padding_grid=[5.0, 10.0],
        baseline_band=(0.35, 0.45),
        baseline_window_padding_s=10.0,
        scenario_map=_small_scenarios(),
        max_points_for_bb=300,
        max_points_for_sig=300,
    )

    assert not candidate_df.empty
    assert not best_df.empty
    assert set(
        [
            "candidate_id",
            "freq_band_min",
            "freq_band_max",
            "window_padding_s",
            "tpr_sig_target_mean",
            "fpr_sig_target_mean",
            "score_tpr_minus_fpr",
            "passes_balanced_bar",
            "delta_tpr_vs_baseline",
            "delta_fpr_vs_baseline",
            "is_baseline",
            "rank",
        ]
    ).issubset(set(candidate_df.columns))

    baseline = candidate_df[candidate_df["is_baseline"]]
    assert len(baseline) == 1
    assert np.isclose(float(baseline["delta_tpr_vs_baseline"].iloc[0]), 0.0, atol=1e-12)
    assert np.isclose(float(baseline["delta_fpr_vs_baseline"].iloc[0]), 0.0, atol=1e-12)
    assert int(best_df["rank"].iloc[0]) == 1

    assert (out_dir / "candidate_summary.csv").exists()
    assert (out_dir / "best_candidate.csv").exists()
    assert (fig_dir / "sprint4_window_band_score.png").exists()
    assert (fig_dir / "sprint4_window_band_tpr_fpr.png").exists()
    assert log_path.exists()

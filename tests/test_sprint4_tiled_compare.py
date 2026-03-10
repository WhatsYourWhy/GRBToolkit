from __future__ import annotations

from grb_refresh import BenchmarkConfig, SimulationParams
from run_sprint4_tiled_compare import run_sprint4_tiled_compare


def _small_scenarios() -> dict[str, SimulationParams]:
    return {
        "mid": SimulationParams(
            A1=650.0,
            tau1=10.0,
            tau_r=1.4,
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
            seed=9101,
        ),
        "boat_drift": SimulationParams(
            A1=950.0,
            tau1=18.0,
            tau_r=2.2,
            B=0.3,
            f_qpo=0.41,
            phi1=0.1,
            N=60,
            Ai=2.0,
            R_bg=25.0,
            t0=0.0,
            k=0.00001,
            T=16.0,
            dt=0.03,
            seed=9102,
            Gamma=True,
        ),
    }


def test_run_sprint4_tiled_compare_smoke(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("mid", "boat_drift"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=2,
        n_surrogates=20,
        alpha=0.05,
        seed_start=900000,
        tile_window_s=4.0,
        tile_step_s=2.0,
        tile_correction_method="bh",
        tile_min_points=32,
        tile_max_windows=6,
    )

    out_dir = tmp_path / "outputs" / "sprint4_tiled_compare"
    fig_dir = tmp_path / "figures"
    log_path = tmp_path / "docs" / "sprint4_tiled_log.md"

    comparison_df, decision_df = run_sprint4_tiled_compare(
        config=cfg,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        scenario_map=_small_scenarios(),
        max_points_for_bb=300,
        max_points_for_sig=300,
    )

    assert not comparison_df.empty
    assert not decision_df.empty
    assert set(comparison_df["detector_variant"].unique().tolist()) == {"windowed_fft_sig", "tiled_window_fft_sig"}
    assert "target_group" in comparison_df["scenario"].unique().tolist()
    assert "improves_target_score" in decision_df.columns

    assert (out_dir / "run_level_windowed_fft_sig.csv").exists()
    assert (out_dir / "summary_windowed_fft_sig.csv").exists()
    assert (out_dir / "run_level_tiled_window_fft_sig.csv").exists()
    assert (out_dir / "summary_tiled_window_fft_sig.csv").exists()
    assert (out_dir / "tiled_comparison.csv").exists()
    assert (out_dir / "tiled_decision.csv").exists()
    assert (fig_dir / "sprint4_tiled_tradeoff.png").exists()
    assert log_path.exists()

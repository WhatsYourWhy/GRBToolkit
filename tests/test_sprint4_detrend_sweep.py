from __future__ import annotations

from grb_refresh import BenchmarkConfig, SimulationParams
from run_sprint4_detrend_sweep import run_sprint4_detrend_sweep


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
            seed=11101,
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
            seed=11102,
            Gamma=True,
        ),
    }


def test_run_sprint4_detrend_sweep_smoke(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("mid", "boat_drift"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=2,
        n_surrogates=20,
        alpha=0.05,
        seed_start=930000,
    )

    out_dir = tmp_path / "outputs" / "sprint4_detrend_sweep"
    fig_dir = tmp_path / "figures"
    log_path = tmp_path / "docs" / "sprint4_detrend_log.md"

    candidate_df, decision_df = run_sprint4_detrend_sweep(
        config=cfg,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        detrend_order_grid=(1, 2),
        holdout_seed_offset=10000,
        min_delta_tpr=0.01,
        max_fpr_target=0.1,
        min_delta_score=0.0,
        scenario_map=_small_scenarios(),
        max_points_for_bb=300,
        max_points_for_sig=300,
    )

    assert not candidate_df.empty
    assert not decision_df.empty
    assert {"train", "holdout"} == set(candidate_df["split"].unique().tolist())
    assert "recommended_action" in decision_df.columns
    assert decision_df["recommended_action"].iloc[0] in {"GO_KEEP_DETREND", "TUNE_MORE", "STOP_OR_PIVOT"}
    assert set(candidate_df["detector_variant"].unique().tolist()) == {"windowed_fft_sig", "detrended_fft_sig"}

    assert (out_dir / "detrend_candidate_summary.csv").exists()
    assert (out_dir / "detrend_best_candidate.csv").exists()
    assert (out_dir / "detrend_decision.csv").exists()
    assert (out_dir / "run_level_train_baseline.csv").exists()
    assert (out_dir / "summary_train_baseline.csv").exists()
    assert (out_dir / "run_level_holdout_baseline.csv").exists()
    assert (out_dir / "summary_holdout_baseline.csv").exists()
    assert (fig_dir / "sprint4_detrend_tradeoff.png").exists()
    assert (fig_dir / "sprint4_detrend_delta.png").exists()
    assert log_path.exists()


def test_run_sprint4_detrend_sweep_resume_reuses_existing_outputs(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("mid", "boat_drift"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=1,
        n_surrogates=10,
        alpha=0.05,
        seed_start=940000,
    )

    out_dir = tmp_path / "outputs" / "sprint4_detrend_sweep"
    fig_dir = tmp_path / "figures"
    log_path = tmp_path / "docs" / "sprint4_detrend_log.md"

    run_sprint4_detrend_sweep(
        config=cfg,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        detrend_order_grid=(1,),
        holdout_seed_offset=10000,
        min_delta_tpr=0.01,
        max_fpr_target=0.1,
        min_delta_score=0.0,
        scenario_map=_small_scenarios(),
        max_points_for_bb=240,
        max_points_for_sig=240,
    )

    baseline_csv = out_dir / "run_level_train_baseline.csv"
    candidate_csv = out_dir / "run_level_train_D00.csv"
    baseline_mtime_before = baseline_csv.stat().st_mtime
    candidate_mtime_before = candidate_csv.stat().st_mtime

    run_sprint4_detrend_sweep(
        config=cfg,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        detrend_order_grid=(1,),
        holdout_seed_offset=10000,
        min_delta_tpr=0.01,
        max_fpr_target=0.1,
        min_delta_score=0.0,
        scenario_map=_small_scenarios(),
        max_points_for_bb=240,
        max_points_for_sig=240,
    )

    assert baseline_csv.stat().st_mtime == baseline_mtime_before
    assert candidate_csv.stat().st_mtime == candidate_mtime_before

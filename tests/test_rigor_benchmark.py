import numpy as np
import pandas as pd

from grb_refresh import BenchmarkConfig, SimulationParams, detect_qpo_signal, summarize_rigor_results
from run_rigor_benchmark import run_rigor_benchmark


def _small_scenarios() -> dict[str, SimulationParams]:
    return {
        "short_70": SimulationParams(
            A1=120.0,
            tau1=0.5,
            tau_r=0.1,
            B=0.3,
            f_qpo=0.41,
            phi1=0.0,
            N=12,
            Ai=6.0,
            R_bg=10.0,
            t0=0.0,
            k=0.0,
            T=3.0,
            dt=0.01,
            seed=10,
        ),
        "boat_drift": SimulationParams(
            A1=220.0,
            tau1=8.0,
            tau_r=1.2,
            B=0.3,
            f_qpo=0.41,
            phi1=0.2,
            N=120,
            Ai=2.0,
            R_bg=15.0,
            t0=0.0,
            k=0.00001,
            T=10.0,
            dt=0.02,
            seed=20,
        ),
    }


def test_detect_qpo_signal_rule():
    assert detect_qpo_signal(0.409, 0.41, 0.02)
    assert not detect_qpo_signal(0.45, 0.41, 0.02)
    assert not detect_qpo_signal(float("nan"), 0.41, 0.02)


def test_summarize_rigor_results_aggregation():
    run_df = pd.DataFrame(
        [
            {
                "scenario": "weak",
                "B": 0.0,
                "p0_scale": 1.0,
                "seed": 1,
                "knots": 10,
                "edge_pct": 5.0,
                "f0_est": 0.41,
                "detected": True,
                "residual_qpo": 4.0,
                "residual_fred": 5.0,
            },
            {
                "scenario": "weak",
                "B": 0.0,
                "p0_scale": 1.0,
                "seed": 2,
                "knots": 14,
                "edge_pct": 7.0,
                "f0_est": 0.39,
                "detected": False,
                "residual_qpo": 3.0,
                "residual_fred": 6.0,
            },
        ]
    )
    summary = summarize_rigor_results(run_df)
    row = summary.iloc[0]

    assert np.isclose(row["recovery_rate"], 0.5)
    assert np.isclose(row["false_positive_rate"], 0.5)
    assert np.isclose(row["median_knots"], 12.0)
    assert np.isclose(row["iqr_knots"], 2.0)
    assert np.isclose(row["median_edge_pct"], 6.0)


def test_benchmark_runner_smoke_outputs(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("short_70", "boat_drift"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=5,
        freq_tolerance_hz=0.02,
        seed_start=120000,
    )

    out_dir = tmp_path / "outputs"
    fig_dir = tmp_path / "figures"
    paper_path = tmp_path / "paper" / "grb_substructure_v2.md"

    run_df, summary_df = run_rigor_benchmark(
        config=cfg,
        output_dir=out_dir,
        figures_dir=fig_dir,
        paper_path=paper_path,
        scenario_map=_small_scenarios(),
        max_points_for_bb=500,
    )

    assert set(run_df.columns) == {
        "scenario",
        "B",
        "p0_scale",
        "seed",
        "knots",
        "edge_pct",
        "f0_est",
        "detected",
        "residual_qpo",
        "residual_fred",
    }
    assert set(summary_df.columns) == {
        "scenario",
        "B",
        "p0_scale",
        "n_runs",
        "recovery_rate",
        "false_positive_rate",
        "median_knots",
        "iqr_knots",
        "median_edge_pct",
        "median_residual_gain_pct",
    }

    assert (out_dir / "run_level_results.csv").exists()
    assert (out_dir / "recovery_summary.csv").exists()
    assert (fig_dir / "recovery_heatmap.png").exists()
    assert (fig_dir / "fpr_vs_p0.png").exists()
    assert (fig_dir / "knot_stability.png").exists()
    assert paper_path.exists()
    assert "Injection-Recovery and False-Positive Benchmark" in paper_path.read_text(encoding="utf-8")

    b0_rows = summary_df[summary_df["B"] == 0.0]
    assert not b0_rows.empty
    assert b0_rows["false_positive_rate"].notna().all()
    assert ((b0_rows["false_positive_rate"] >= 0.0) & (b0_rows["false_positive_rate"] <= 1.0)).all()


def test_benchmark_slice_is_deterministic(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("short_70",),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=3,
        freq_tolerance_hz=0.02,
        seed_start=130000,
    )
    scenarios = {"short_70": _small_scenarios()["short_70"]}

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    fig_a = tmp_path / "fig_a"
    fig_b = tmp_path / "fig_b"
    paper_a = tmp_path / "paper_a.md"
    paper_b = tmp_path / "paper_b.md"

    run_a, summary_a = run_rigor_benchmark(
        config=cfg,
        output_dir=out_a,
        figures_dir=fig_a,
        paper_path=paper_a,
        scenario_map=scenarios,
        max_points_for_bb=400,
    )
    run_b, summary_b = run_rigor_benchmark(
        config=cfg,
        output_dir=out_b,
        figures_dir=fig_b,
        paper_path=paper_b,
        scenario_map=scenarios,
        max_points_for_bb=400,
    )

    pd.testing.assert_frame_equal(run_a.reset_index(drop=True), run_b.reset_index(drop=True))
    pd.testing.assert_frame_equal(summary_a.reset_index(drop=True), summary_b.reset_index(drop=True))

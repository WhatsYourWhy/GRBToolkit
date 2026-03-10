import numpy as np
import pandas as pd

from grb_refresh import (
    BenchmarkConfig,
    SimulationParams,
    detect_qpo_signal,
    estimate_surrogate_p_value,
    phase_randomized_surrogate,
    summarize_rigor_results,
)
from run_rigor_benchmark import run_rigor_benchmark
from run_rigor_benchmark import _adjust_pvalues
from run_rigor_benchmark import _compute_welch_band_peak
from run_rigor_benchmark import _detrend_signal


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
        "boat_transient": SimulationParams(
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
            qpo_window_start=3.0,
            qpo_window_end=7.0,
        ),
    }


def test_detect_qpo_signal_rule():
    assert detect_qpo_signal(0.409, 0.41, 0.02)
    assert not detect_qpo_signal(0.45, 0.41, 0.02)
    assert not detect_qpo_signal(float("nan"), 0.41, 0.02)


def test_phase_randomized_surrogate_preserves_amplitude_and_changes_phase():
    rng = np.random.default_rng(1234)
    x = rng.normal(0.0, 1.0, size=256)
    s = phase_randomized_surrogate(x, rng)

    X = np.fft.rfft(x)
    S = np.fft.rfft(s)

    np.testing.assert_allclose(np.abs(X), np.abs(S), rtol=1e-10, atol=1e-10)
    assert not np.allclose(np.angle(X[1:-1]), np.angle(S[1:-1]))


def test_surrogate_p_value_bounds_and_monotonicity():
    sur = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    p_low = estimate_surrogate_p_value(5.0, sur)
    p_high = estimate_surrogate_p_value(1.5, sur)

    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_low < p_high


def test_detrend_signal_preserves_shape_and_reduces_linear_component():
    x = np.linspace(0.0, 5.0, 200)
    signal = 3.0 * x + 0.5 * np.sin(2.0 * np.pi * 0.4 * x)
    detrended = _detrend_signal(signal, dt=float(x[1] - x[0]), order=1)

    assert detrended.shape == signal.shape
    assert np.isfinite(detrended).all()
    coeff_after = np.polyfit(x, detrended, deg=1)
    assert abs(coeff_after[0]) < 0.2


def test_welch_band_peak_finds_injected_frequency():
    rng = np.random.default_rng(123)
    dt = 0.02
    t = np.arange(0.0, 20.0, dt)
    f_true = 0.41
    signal = np.sin(2.0 * np.pi * f_true * t) + 0.2 * rng.normal(0.0, 1.0, size=t.size)
    peak_power, peak_freq = _compute_welch_band_peak(
        signal=signal,
        dt=dt,
        fmin=0.35,
        fmax=0.45,
        segment_points=256,
        overlap_frac=0.5,
        use_hann=True,
    )

    assert np.isfinite(peak_power)
    assert np.isfinite(peak_freq)
    assert abs(float(peak_freq) - f_true) <= 0.03


def test_adjust_pvalues_bh_and_bonferroni_are_bounded():
    raw = np.array([0.01, 0.03, 0.2], dtype=np.float64)
    bh = _adjust_pvalues(raw, method="bh")
    bonf = _adjust_pvalues(raw, method="bonferroni")

    assert np.all(np.isfinite(bh))
    assert np.all(np.isfinite(bonf))
    assert np.all((bh >= 0.0) & (bh <= 1.0))
    assert np.all((bonf >= 0.0) & (bonf <= 1.0))
    assert np.all(bonf >= raw)


def test_summarize_rigor_results_aggregation_and_ci():
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
                "detected_hit": True,
                "detected_sig": False,
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
                "detected_hit": False,
                "detected_sig": False,
                "residual_qpo": 3.0,
                "residual_fred": 6.0,
            },
        ]
    )
    summary = summarize_rigor_results(run_df)
    row = summary.iloc[0]

    assert np.isclose(row["recovery_rate"], 0.5)
    assert np.isclose(row["false_positive_rate"], 0.5)
    assert np.isclose(row["recovery_rate_sig"], 0.0)
    assert np.isclose(row["false_positive_rate_sig"], 0.0)
    assert np.isclose(row["median_knots"], 12.0)
    assert np.isclose(row["iqr_knots"], 2.0)
    assert np.isclose(row["median_edge_pct"], 6.0)
    assert 0.0 <= row["recovery_rate_sig_ci_low"] <= row["recovery_rate_sig_ci_high"] <= 1.0


def test_benchmark_runner_smoke_outputs_and_modes(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("short_70", "boat_transient"),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=5,
        n_surrogates=50,
        freq_tolerance_hz=0.02,
        seed_start=120000,
        detector_variant="windowed_fft_sig",
        detrend_order=1,
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
        max_points_for_sig=500,
    )

    expected_run_cols = {
        "scenario",
        "B",
        "p0_scale",
        "seed",
        "knots",
        "edge_pct",
        "f0_est",
        "detected",
        "detected_hit",
        "residual_qpo",
        "residual_fred",
        "peak_power_obs",
        "peak_freq_obs",
        "p_value",
        "detected_sig",
        "detection_mode",
        "p_value_global",
        "p_value_window",
        "detector_variant",
        "detrend_order",
        "peak_statistic",
    }
    assert expected_run_cols.issubset(set(run_df.columns))
    assert "recovery_rate_sig" in summary_df.columns
    assert "false_positive_rate_sig" in summary_df.columns
    assert "recovery_rate_sig_ci_low" in summary_df.columns
    assert "recovery_rate_sig_ci_high" in summary_df.columns
    assert "false_positive_rate_sig_ci_low" in summary_df.columns
    assert "false_positive_rate_sig_ci_high" in summary_df.columns

    assert (out_dir / "run_level_results.csv").exists()
    assert (out_dir / "recovery_summary.csv").exists()
    assert (fig_dir / "recovery_heatmap_sig.png").exists()
    assert (fig_dir / "fpr_vs_p0_sig.png").exists()
    assert (fig_dir / "knot_stability.png").exists()
    assert (fig_dir / "pvalue_distribution.png").exists()
    assert paper_path.exists()
    assert "Injection-Recovery and False-Positive Benchmark" in paper_path.read_text(encoding="utf-8")

    b0_rows = summary_df[summary_df["B"] == 0.0]
    assert not b0_rows.empty
    assert b0_rows["false_positive_rate_sig"].notna().all()
    assert ((b0_rows["false_positive_rate_sig"] >= 0.0) & (b0_rows["false_positive_rate_sig"] <= 1.0)).all()
    assert run_df[run_df["scenario"] == "boat_transient"]["detection_mode"].eq("windowed").all()
    assert run_df[run_df["scenario"] == "short_70"]["detection_mode"].eq("global").all()
    assert run_df["detector_variant"].eq("windowed_fft_sig").all()
    assert run_df["detrend_order"].eq(1).all()
    assert run_df["peak_statistic"].eq("fft").all()
    assert run_df["p_value"].notna().all()
    assert ((run_df["p_value"] >= 0.0) & (run_df["p_value"] <= 1.0)).all()


def test_global_variant_forces_global_mode_even_for_transient(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("boat_transient",),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=3,
        n_surrogates=20,
        seed_start=220000,
        detector_variant="global_tapered_fft_sig",
    )

    run_df, _ = run_rigor_benchmark(
        config=cfg,
        output_dir=tmp_path / "outputs",
        figures_dir=tmp_path / "figures",
        paper_path=tmp_path / "paper" / "grb_substructure_v2.md",
        scenario_map={"boat_transient": _small_scenarios()["boat_transient"]},
        max_points_for_bb=300,
        max_points_for_sig=300,
    )

    assert run_df["detection_mode"].eq("global").all()


def test_benchmark_slice_is_deterministic(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("short_70",),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=3,
        n_surrogates=20,
        freq_tolerance_hz=0.02,
        seed_start=130000,
        detector_variant="detrended_fft_sig",
        detrend_order=1,
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
        max_points_for_sig=400,
    )
    run_b, summary_b = run_rigor_benchmark(
        config=cfg,
        output_dir=out_b,
        figures_dir=fig_b,
        paper_path=paper_b,
        scenario_map=scenarios,
        max_points_for_bb=400,
        max_points_for_sig=400,
    )

    pd.testing.assert_frame_equal(run_a.reset_index(drop=True), run_b.reset_index(drop=True))
    pd.testing.assert_frame_equal(summary_a.reset_index(drop=True), summary_b.reset_index(drop=True))


def test_welch_variant_smoke_sets_peak_statistic(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("short_70",),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=3,
        n_surrogates=20,
        seed_start=230000,
        detector_variant="welch_fft_sig",
        welch_segment_points=128,
        welch_overlap_frac=0.5,
    )

    run_df, _ = run_rigor_benchmark(
        config=cfg,
        output_dir=tmp_path / "outputs",
        figures_dir=tmp_path / "figures",
        paper_path=tmp_path / "paper" / "grb_substructure_v2.md",
        scenario_map={"short_70": _small_scenarios()["short_70"]},
        max_points_for_bb=300,
        max_points_for_sig=300,
    )

    assert run_df["peak_statistic"].eq("welch").all()


def test_tiled_variant_smoke_sets_mode_and_pvalues(tmp_path):
    cfg = BenchmarkConfig(
        scenario_names=("boat_transient",),
        b_grid=(0.0, 0.3),
        p0_scale_grid=(1.0,),
        n_replicates=3,
        n_surrogates=20,
        seed_start=240000,
        detector_variant="tiled_window_fft_sig",
        tile_window_s=2.0,
        tile_step_s=1.0,
        tile_correction_method="bh",
        tile_min_points=32,
        tile_max_windows=6,
    )

    run_df, _ = run_rigor_benchmark(
        config=cfg,
        output_dir=tmp_path / "outputs",
        figures_dir=tmp_path / "figures",
        paper_path=tmp_path / "paper" / "grb_substructure_v2.md",
        scenario_map={"boat_transient": _small_scenarios()["boat_transient"]},
        max_points_for_bb=300,
        max_points_for_sig=300,
    )

    assert run_df["detection_mode"].eq("tiled").all()
    assert run_df["peak_statistic"].eq("fft").all()
    assert run_df["p_value_tiled_adj"].notna().all()
    assert ((run_df["p_value_tiled_adj"] >= 0.0) & (run_df["p_value_tiled_adj"] <= 1.0)).all()
    assert run_df["tiled_n_windows"].ge(1).all()

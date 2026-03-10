from __future__ import annotations

import numpy as np
import pandas as pd

from run_sprint5_tte_bridge import run_sprint5_tte_bridge


def test_run_sprint5_tte_bridge_csv_smoke(tmp_path):
    rng = np.random.default_rng(1234)
    dt = 0.05
    t = np.arange(0.0, 30.0, dt, dtype=np.float64)
    signal = 12.0 + 2.0 * np.sin(2.0 * np.pi * 0.4 * t) + rng.normal(0.0, 0.7, size=t.size)
    signal = np.clip(signal, 0.0, None)

    burst_csv = tmp_path / "burst.csv"
    pd.DataFrame({"time": t, "signal": signal}).to_csv(burst_csv, index=False)

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "burst_id": "test_burst",
                "input_type": "csv",
                "input_path": str(burst_csv),
                "bin_width_s": dt,
                "qpo_window_start": 8.0,
                "qpo_window_end": 20.0,
                "window_padding_s": 2.0,
            }
        ]
    ).to_csv(manifest_csv, index=False)

    out_dir = tmp_path / "outputs" / "sprint5_tte_bridge"
    fig_dir = tmp_path / "figures"
    log_path = tmp_path / "docs" / "sprint5_tte_bridge_log.md"

    run_df, summary_df = run_sprint5_tte_bridge(
        manifest_path=manifest_csv,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        alpha=0.05,
        n_surrogates=20,
        n_null_trials=10,
        freq_band_min=0.35,
        freq_band_max=0.45,
        default_bin_width_s=dt,
        default_window_padding_s=2.0,
        max_points_for_sig=1024,
        seed=880000,
        update_paper=False,
    )

    assert not run_df.empty
    assert not summary_df.empty
    assert run_df.loc[0, "burst_id"] == "test_burst"
    assert run_df.loc[0, "detection_mode"] == "windowed"
    assert "null_empirical_fpr_alpha" in run_df.columns
    assert "calibration_status" in summary_df.columns

    assert (out_dir / "tte_bridge_results.csv").exists()
    assert (out_dir / "tte_bridge_summary.csv").exists()
    assert (fig_dir / "sprint5_tte_pvalues.png").exists()
    assert (fig_dir / "sprint5_tte_null_calibration.png").exists()
    assert log_path.exists()

from __future__ import annotations

import numpy as np
import pandas as pd

from run_sprint5_tte_solidify import run_sprint5_tte_solidify


def test_run_sprint5_tte_solidify_smoke(tmp_path):
    rng = np.random.default_rng(2222)
    dt = 0.05
    t = np.arange(0.0, 20.0, dt, dtype=np.float64)
    signal = 9.0 + 1.8 * np.sin(2.0 * np.pi * 0.4 * t) + rng.normal(0.0, 0.8, size=t.size)
    signal = np.clip(signal, 0.0, None)

    burst_csv = tmp_path / "burst.csv"
    pd.DataFrame({"time": t, "signal": signal}).to_csv(burst_csv, index=False)

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "burst_id": "solidify_burst",
                "input_type": "csv",
                "input_path": str(burst_csv),
            }
        ]
    ).to_csv(manifest_csv, index=False)

    out_dir = tmp_path / "outputs" / "solidify"
    fig_dir = tmp_path / "figures"
    log_path = tmp_path / "docs" / "sprint5_tte_solidify_log.md"

    summary_df, burst_df, decision_df = run_sprint5_tte_solidify(
        manifest_path=manifest_csv,
        output_dir=out_dir,
        figures_dir=fig_dir,
        log_path=log_path,
        seed_grid=(701000, 701500),
        bin_width_grid_s=(0.05,),
        band_grid=((0.35, 0.45),),
        alpha=0.05,
        n_surrogates=20,
        n_null_trials=10,
        default_window_padding_s=2.0,
        max_points_for_sig=1024,
    )

    assert not summary_df.empty
    assert not burst_df.empty
    assert not decision_df.empty
    assert set(summary_df["run_id"].tolist()) == {"R000", "R001"}
    assert "recommended_action" in decision_df.columns

    assert (out_dir / "solidify_summary_matrix.csv").exists()
    assert (out_dir / "solidify_burst_matrix.csv").exists()
    assert (out_dir / "solidify_decision.csv").exists()
    assert (fig_dir / "sprint5_tte_solidify_detected_fraction.png").exists()
    assert (fig_dir / "sprint5_tte_solidify_null_fpr.png").exists()
    assert log_path.exists()

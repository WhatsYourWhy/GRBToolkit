from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_sprint5_tte_bridge import run_sprint5_tte_bridge


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_band_grid(raw: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Band token '{token}' must use min:max format")
        lo_raw, hi_raw = token.split(":", 1)
        lo = float(lo_raw)
        hi = float(hi_raw)
        if lo >= hi:
            raise ValueError(f"Band token '{token}' must satisfy min < max")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("Band grid cannot be empty")
    return bands


def _format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append("nan" if np.isnan(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plot_heatmap(
    matrix_df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    if matrix_df.empty:
        return
    pivot = (
        matrix_df.pivot_table(
            index="bin_width_s",
            columns="band_label",
            values=value_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{val:.3f}" for val in pivot.index.to_numpy(dtype=float)])
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("Bin width (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_decision(summary_matrix_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if summary_matrix_df.empty:
        return pd.DataFrame(
            [
                {
                    "n_runs": 0,
                    "n_runs_with_detection": 0,
                    "max_detected_fraction": np.nan,
                    "mean_detected_fraction": np.nan,
                    "max_null_empirical_fpr_alpha": np.nan,
                    "mean_null_empirical_fpr_alpha": np.nan,
                    "alpha": float(alpha),
                    "stable_non_detection": False,
                    "stable_calibration": False,
                    "recommended_action": "INSUFFICIENT_DATA",
                    "reason": "No sweep runs were evaluated.",
                }
            ]
        )

    n_runs = int(len(summary_matrix_df))
    n_detection = int((summary_matrix_df["n_detected_sig"] > 0).sum())
    max_det_frac = float(summary_matrix_df["detected_fraction"].max())
    mean_det_frac = float(summary_matrix_df["detected_fraction"].mean())
    max_null = float(summary_matrix_df["mean_null_empirical_fpr_alpha"].max())
    mean_null = float(summary_matrix_df["mean_null_empirical_fpr_alpha"].mean())

    stable_non_detection = bool(n_detection == 0 and np.isfinite(max_det_frac) and max_det_frac == 0.0)
    stable_calibration = bool(np.isfinite(max_null) and max_null <= float(alpha) * 1.5)

    if stable_non_detection and stable_calibration:
        action = "LOCK_MIXED_NEGATIVE_METHODS_CLAIM"
        reason = "No significance detections across sweep and null calibration remained controlled."
    elif stable_non_detection and not stable_calibration:
        action = "TIGHTEN_CALIBRATION_BEFORE_CLAIMS"
        reason = "Detections remained absent, but null calibration drifted high in at least one sweep cell."
    else:
        action = "INVESTIGATE_CONFIG_SENSITIVE_SIGNALS"
        reason = "At least one sweep cell showed significance detections; inspect burst-level behavior before framing."

    return pd.DataFrame(
        [
            {
                "n_runs": n_runs,
                "n_runs_with_detection": n_detection,
                "max_detected_fraction": max_det_frac,
                "mean_detected_fraction": mean_det_frac,
                "max_null_empirical_fpr_alpha": max_null,
                "mean_null_empirical_fpr_alpha": mean_null,
                "alpha": float(alpha),
                "stable_non_detection": stable_non_detection,
                "stable_calibration": stable_calibration,
                "recommended_action": action,
                "reason": reason,
            }
        ]
    )


def _write_log(log_path: Path, top_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Sprint 5 TTE Solidify Log

## Sweep Summary (Top 12 by detected fraction, then calibration)
{_format_markdown_table(top_df)}

## Decision
{_format_markdown_table(decision_df)}

## Interpretation Guardrail
- This sweep is a methods-validation robustness check.
- Use these results to calibrate claims and uncertainty language, not to assert astrophysical detection.
"""
    log_path.write_text(text, encoding="utf-8")


def run_sprint5_tte_solidify(
    manifest_path: Path,
    output_dir: Path,
    figures_dir: Path,
    log_path: Path,
    seed_grid: Sequence[int],
    bin_width_grid_s: Sequence[float],
    band_grid: Sequence[tuple[float, float]],
    alpha: float = 0.05,
    n_surrogates: int = 200,
    n_null_trials: int = 80,
    default_window_padding_s: float = 10.0,
    max_points_for_sig: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not seed_grid:
        raise ValueError("seed_grid cannot be empty")
    if not bin_width_grid_s:
        raise ValueError("bin_width_grid_s cannot be empty")
    if not band_grid:
        raise ValueError("band_grid cannot be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    burst_rows: list[pd.DataFrame] = []
    run_idx = 0
    for seed in seed_grid:
        for bin_width_s in bin_width_grid_s:
            for band_lo, band_hi in band_grid:
                run_id = f"R{run_idx:03d}"
                run_idx += 1
                band_label = f"{band_lo:.2f}-{band_hi:.2f}"
                run_dir = output_dir / run_id
                run_fig_dir = figures_dir / "_sprint5_tte_solidify_tmp" / run_id
                run_log_path = output_dir / f"log_{run_id}.md"

                run_df, summary_df = run_sprint5_tte_bridge(
                    manifest_path=manifest_path,
                    output_dir=run_dir,
                    figures_dir=run_fig_dir,
                    log_path=run_log_path,
                    alpha=float(alpha),
                    n_surrogates=int(n_surrogates),
                    n_null_trials=int(n_null_trials),
                    freq_band_min=float(band_lo),
                    freq_band_max=float(band_hi),
                    default_bin_width_s=float(bin_width_s),
                    default_window_padding_s=float(default_window_padding_s),
                    max_points_for_sig=int(max_points_for_sig),
                    seed=int(seed),
                    force_bin_width_s=float(bin_width_s),
                    update_paper=False,
                )

                run_df = run_df.copy()
                run_df["run_id"] = run_id
                run_df["seed"] = int(seed)
                run_df["bin_width_s_cfg"] = float(bin_width_s)
                run_df["freq_band_min_cfg"] = float(band_lo)
                run_df["freq_band_max_cfg"] = float(band_hi)
                run_df["band_label"] = band_label
                burst_rows.append(run_df)

                summary_row = summary_df.iloc[0].to_dict()
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "seed": int(seed),
                        "bin_width_s": float(bin_width_s),
                        "freq_band_min": float(band_lo),
                        "freq_band_max": float(band_hi),
                        "band_label": band_label,
                        "n_bursts": int(summary_row["n_bursts"]),
                        "n_detected_sig": int(summary_row["n_detected_sig"]),
                        "detected_fraction": float(summary_row["detected_fraction"]),
                        "median_p_value": float(summary_row["median_p_value"]),
                        "mean_null_empirical_fpr_alpha": float(summary_row["mean_null_empirical_fpr_alpha"]),
                        "calibration_status": str(summary_row["calibration_status"]),
                    }
                )

    summary_matrix_df = pd.DataFrame(summary_rows).sort_values(
        ["detected_fraction", "mean_null_empirical_fpr_alpha"],
        ascending=[False, True],
    ).reset_index(drop=True)
    burst_matrix_df = pd.concat(burst_rows, ignore_index=True) if burst_rows else pd.DataFrame()
    decision_df = _build_decision(summary_matrix_df=summary_matrix_df, alpha=float(alpha))

    summary_csv = output_dir / "solidify_summary_matrix.csv"
    burst_csv = output_dir / "solidify_burst_matrix.csv"
    decision_csv = output_dir / "solidify_decision.csv"
    summary_matrix_df.to_csv(summary_csv, index=False)
    burst_matrix_df.to_csv(burst_csv, index=False)
    decision_df.to_csv(decision_csv, index=False)

    det_heatmap = figures_dir / "sprint5_tte_solidify_detected_fraction.png"
    cal_heatmap = figures_dir / "sprint5_tte_solidify_null_fpr.png"
    _plot_heatmap(
        matrix_df=summary_matrix_df,
        value_col="detected_fraction",
        title="Sprint 5 Solidify: Detected Fraction",
        cbar_label="Detected fraction",
        output_path=det_heatmap,
        vmin=0.0,
        vmax=1.0,
    )
    _plot_heatmap(
        matrix_df=summary_matrix_df,
        value_col="mean_null_empirical_fpr_alpha",
        title="Sprint 5 Solidify: Mean Null Empirical FPR",
        cbar_label="Mean null empirical FPR",
        output_path=cal_heatmap,
        vmin=0.0,
        vmax=max(1.0, float(summary_matrix_df["mean_null_empirical_fpr_alpha"].max()) * 1.1),
    )

    top_df = summary_matrix_df.head(12).copy()
    _write_log(log_path=log_path, top_df=top_df, decision_df=decision_df)

    print("[sprint5-tte-solidify] complete")
    print(f"[sprint5-tte-solidify] summary matrix: {summary_csv}")
    print(f"[sprint5-tte-solidify] burst matrix: {burst_csv}")
    print(f"[sprint5-tte-solidify] decision: {decision_csv}")
    print(f"[sprint5-tte-solidify] figures: {det_heatmap}, {cal_heatmap}")
    print(f"[sprint5-tte-solidify] log: {log_path}")
    return summary_matrix_df, burst_matrix_df, decision_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 5 robustness sweep for the real-TTE bridge pilot.")
    parser.add_argument("--manifest-path", default="docs/sprint5_tte_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/sprint5_tte_solidify")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--log-path", default="docs/sprint5_tte_solidify_log.md")
    parser.add_argument("--seeds", default="701000,701500,702000")
    parser.add_argument("--bin-width-grid-s", default="0.02,0.05,0.1")
    parser.add_argument("--bands", default="0.30:0.40,0.35:0.45")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-surrogates", type=int, default=200)
    parser.add_argument("--n-null-trials", type=int, default=80)
    parser.add_argument("--default-window-padding-s", type=float, default=10.0)
    parser.add_argument("--max-points-for-sig", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sprint5_tte_solidify(
        manifest_path=Path(args.manifest_path),
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        log_path=Path(args.log_path),
        seed_grid=tuple(_parse_int_list(args.seeds)),
        bin_width_grid_s=tuple(_parse_float_list(args.bin_width_grid_s)),
        band_grid=tuple(_parse_band_grid(args.bands)),
        alpha=float(args.alpha),
        n_surrogates=int(args.n_surrogates),
        n_null_trials=int(args.n_null_trials),
        default_window_padding_s=float(args.default_window_padding_s),
        max_points_for_sig=int(args.max_points_for_sig),
    )

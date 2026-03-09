from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grb_refresh import BenchmarkConfig, get_default_scenarios
from run_rigor_benchmark import run_rigor_benchmark

TARGET_SCENARIOS = ("mid", "boat_drift")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_band_grid(raw: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for token in [item.strip() for item in raw.split(",") if item.strip()]:
        if ":" not in token:
            raise ValueError(f"Band token '{token}' must use 'min:max' format.")
        lo_raw, hi_raw = token.split(":", 1)
        lo = float(lo_raw)
        hi = float(hi_raw)
        if lo >= hi:
            raise ValueError(f"Band token '{token}' must satisfy min < max.")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("Band grid cannot be empty.")
    return bands


def _target_metrics(summary_df: pd.DataFrame) -> tuple[float, float, bool]:
    scenario_rows = summary_df[summary_df["scenario"].isin(TARGET_SCENARIOS)].copy()
    if scenario_rows.empty:
        return float("nan"), float("nan"), False

    by_scenario = []
    for scenario in TARGET_SCENARIOS:
        chunk = scenario_rows[scenario_rows["scenario"] == scenario]
        if chunk.empty:
            by_scenario.append((float("nan"), float("nan"), False))
            continue

        tpr = float(chunk.loc[chunk["B"] >= 0.2, "recovery_rate_sig"].mean())
        fpr = float(chunk.loc[chunk["B"] == 0.0, "false_positive_rate_sig"].mean())
        passes = bool(np.isfinite(tpr) and np.isfinite(fpr) and (tpr >= 0.6) and (fpr <= 0.1))
        by_scenario.append((tpr, fpr, passes))

    tpr_mean = float(np.nanmean([row[0] for row in by_scenario]))
    fpr_mean = float(np.nanmean([row[1] for row in by_scenario]))
    passes_all = bool(all(row[2] for row in by_scenario))
    return tpr_mean, fpr_mean, passes_all


def _plot_score_heatmap(candidate_df: pd.DataFrame, output_path: Path) -> None:
    if candidate_df.empty:
        return
    work = candidate_df.copy()
    work["band_label"] = work.apply(
        lambda row: f"{row['freq_band_min']:.2f}-{row['freq_band_max']:.2f}",
        axis=1,
    )

    pivot = work.pivot_table(
        index="window_padding_s",
        columns="band_label",
        values="score_tpr_minus_fpr",
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{val:.1f}" for val in pivot.index.to_numpy(dtype=float)])
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("Window padding (s)")
    ax.set_title("Sprint 4 Window/Band Score (TPR - FPR)")
    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_tpr_fpr(candidate_df: pd.DataFrame, output_path: Path) -> None:
    if candidate_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for _, row in candidate_df.iterrows():
        ax.scatter(row["fpr_sig_target_mean"], row["tpr_sig_target_mean"], s=55, alpha=0.8)
        ax.text(
            float(row["fpr_sig_target_mean"]) + 0.002,
            float(row["tpr_sig_target_mean"]) + 0.002,
            str(row["candidate_id"]),
            fontsize=7,
        )
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.6, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Target-group FPR mean")
    ax.set_ylabel("Target-group TPR mean")
    ax.set_title("Sprint 4 Candidates: Target TPR vs FPR")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_log(log_path: Path, baseline_row: pd.Series, best_row: pd.Series, top_df: pd.DataFrame) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(top_df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in top_df.iterrows():
        values: list[str] = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append("nan" if np.isnan(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    top_md = "\n".join(lines)
    text = f"""# Sprint 4 Window/Band Optimization Log

## Baseline
- Candidate: `{baseline_row['candidate_id']}`
- Band: `{baseline_row['freq_band_min']:.2f}-{baseline_row['freq_band_max']:.2f} Hz`
- Window padding: `{baseline_row['window_padding_s']:.1f} s`
- Target mean TPR: `{baseline_row['tpr_sig_target_mean']:.4f}`
- Target mean FPR: `{baseline_row['fpr_sig_target_mean']:.4f}`
- Target mean score (`TPR-FPR`): `{baseline_row['score_tpr_minus_fpr']:.4f}`

## Best Candidate
- Candidate: `{best_row['candidate_id']}`
- Band: `{best_row['freq_band_min']:.2f}-{best_row['freq_band_max']:.2f} Hz`
- Window padding: `{best_row['window_padding_s']:.1f} s`
- Target mean TPR: `{best_row['tpr_sig_target_mean']:.4f}`
- Target mean FPR: `{best_row['fpr_sig_target_mean']:.4f}`
- Delta TPR vs baseline: `{best_row['delta_tpr_vs_baseline']:.4f}`
- Delta FPR vs baseline: `{best_row['delta_fpr_vs_baseline']:.4f}`
- Passes balanced bar: `{bool(best_row['passes_balanced_bar'])}`

## Top Candidates
{top_md}
"""
    log_path.write_text(text, encoding="utf-8")


def run_sprint4_window_band(
    config: BenchmarkConfig,
    output_dir: Path,
    figures_dir: Path,
    log_path: Path,
    band_grid: Sequence[tuple[float, float]],
    window_padding_grid: Sequence[float],
    baseline_band: tuple[float, float] = (0.35, 0.45),
    baseline_window_padding_s: float = 10.0,
    scenario_map: Mapping[str, object] | None = None,
    max_points_for_bb: int = 2000,
    max_points_for_sig: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_map is None:
        scenario_map = get_default_scenarios()

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    candidates: list[tuple[float, float, float]] = []
    for lo, hi in band_grid:
        for padding in window_padding_grid:
            candidates.append((float(lo), float(hi), float(padding)))
    if (baseline_band[0], baseline_band[1], float(baseline_window_padding_s)) not in candidates:
        candidates.insert(0, (baseline_band[0], baseline_band[1], float(baseline_window_padding_s)))

    for idx, (band_lo, band_hi, padding) in enumerate(candidates):
        candidate_id = f"C{idx:02d}"
        candidate_cfg = replace(
            config,
            detector_variant="windowed_fft_sig",
            freq_band_min=float(band_lo),
            freq_band_max=float(band_hi),
            window_padding_s=float(padding),
        )
        candidate_dir = output_dir / "candidates" / candidate_id
        candidate_fig_dir = figures_dir / "_sprint4_tmp" / candidate_id
        run_df, summary_df = run_rigor_benchmark(
            config=candidate_cfg,
            output_dir=candidate_dir,
            figures_dir=candidate_fig_dir,
            paper_path=output_dir / "_scratch_paper.md",
            scenario_map=scenario_map,
            max_points_for_bb=max_points_for_bb,
            max_points_for_sig=max_points_for_sig,
            update_paper=False,
        )
        run_df.to_csv(output_dir / f"run_level_{candidate_id}.csv", index=False)
        summary_df.to_csv(output_dir / f"summary_{candidate_id}.csv", index=False)

        tpr_mean, fpr_mean, passes = _target_metrics(summary_df)
        run_rows.append(
            {
                "candidate_id": candidate_id,
                "freq_band_min": float(band_lo),
                "freq_band_max": float(band_hi),
                "window_padding_s": float(padding),
                "tpr_sig_target_mean": tpr_mean,
                "fpr_sig_target_mean": fpr_mean,
                "score_tpr_minus_fpr": float(tpr_mean - fpr_mean) if np.isfinite(tpr_mean) and np.isfinite(fpr_mean) else np.nan,
                "passes_balanced_bar": bool(passes),
            }
        )

    candidate_df = pd.DataFrame(run_rows)
    baseline_mask = (
        np.isclose(candidate_df["freq_band_min"], baseline_band[0])
        & np.isclose(candidate_df["freq_band_max"], baseline_band[1])
        & np.isclose(candidate_df["window_padding_s"], baseline_window_padding_s)
    )
    if not bool(baseline_mask.any()):
        raise ValueError("Baseline candidate was not evaluated.")

    baseline_row = candidate_df.loc[baseline_mask].iloc[0]
    candidate_df["delta_tpr_vs_baseline"] = candidate_df["tpr_sig_target_mean"] - float(baseline_row["tpr_sig_target_mean"])
    candidate_df["delta_fpr_vs_baseline"] = candidate_df["fpr_sig_target_mean"] - float(baseline_row["fpr_sig_target_mean"])
    candidate_df["is_baseline"] = baseline_mask
    candidate_df = candidate_df.sort_values(
        ["score_tpr_minus_fpr", "tpr_sig_target_mean", "fpr_sig_target_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_df["rank"] = np.arange(1, len(candidate_df) + 1, dtype=int)

    best_df = candidate_df.head(1).copy()

    candidate_csv = output_dir / "candidate_summary.csv"
    best_csv = output_dir / "best_candidate.csv"
    candidate_df.to_csv(candidate_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    heatmap_png = figures_dir / "sprint4_window_band_score.png"
    scatter_png = figures_dir / "sprint4_window_band_tpr_fpr.png"
    _plot_score_heatmap(candidate_df, heatmap_png)
    _plot_tpr_fpr(candidate_df, scatter_png)

    _write_log(log_path, baseline_row=baseline_row, best_row=best_df.iloc[0], top_df=candidate_df.head(10))

    print("[sprint4-window-band] complete")
    print(f"[sprint4-window-band] candidate summary: {candidate_csv}")
    print(f"[sprint4-window-band] best candidate: {best_csv}")
    print(f"[sprint4-window-band] figures: {heatmap_png}, {scatter_png}")
    print(f"[sprint4-window-band] log: {log_path}")
    return candidate_df, best_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 4 Item 1: window and frequency-band optimization sweep.")
    parser.add_argument("--output-dir", default="outputs/sprint4_window_band")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--log-path", default="docs/sprint4_window_band_log.md")
    parser.add_argument("--scenarios", default="short_70,short_100,weak,mid,boat_drift,boat_transient")
    parser.add_argument("--b-grid", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--p0-scales", default="0.5,1.0,2.0")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.02)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--detrend-order", type=int, default=1)
    parser.add_argument("--bands", default="0.30:0.40,0.35:0.45,0.38:0.48")
    parser.add_argument("--window-padding-grid", default="5.0,10.0,20.0")
    parser.add_argument("--baseline-band", default="0.35:0.45")
    parser.add_argument("--baseline-window-padding-s", type=float, default=10.0)
    parser.add_argument("--max-points-for-bb", type=int, default=2000)
    parser.add_argument("--max-points-for-sig", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    baseline_band = _parse_band_grid(args.baseline_band)
    if len(baseline_band) != 1:
        raise ValueError("baseline-band must provide exactly one min:max pair")

    cfg = BenchmarkConfig(
        scenario_names=tuple(_parse_str_list(args.scenarios)),
        b_grid=tuple(_parse_float_list(args.b_grid)),
        p0_scale_grid=tuple(_parse_float_list(args.p0_scales)),
        n_replicates=int(args.n_replicates),
        freq_tolerance_hz=float(args.freq_tolerance_hz),
        seed_start=int(args.seed_start),
        n_surrogates=int(args.n_surrogates),
        alpha=float(args.alpha),
        detrend_order=int(args.detrend_order),
    )

    run_sprint4_window_band(
        config=cfg,
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        log_path=Path(args.log_path),
        band_grid=_parse_band_grid(args.bands),
        window_padding_grid=_parse_float_list(args.window_padding_grid),
        baseline_band=baseline_band[0],
        baseline_window_padding_s=float(args.baseline_window_padding_s),
        max_points_for_bb=int(args.max_points_for_bb),
        max_points_for_sig=int(args.max_points_for_sig),
    )

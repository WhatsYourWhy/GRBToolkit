from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grb_refresh import BenchmarkConfig, get_default_scenarios
from run_rigor_benchmark import run_rigor_benchmark

BASELINE_VARIANT = "windowed_fft_sig"
ITEM2_VARIANT = "welch_fft_sig"
TARGET_SCENARIOS = ("mid", "boat_drift")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _scenario_metrics(summary_df: pd.DataFrame, scenario: str) -> tuple[float, float]:
    chunk = summary_df[summary_df["scenario"] == scenario]
    if chunk.empty:
        return float("nan"), float("nan")
    tpr = float(chunk.loc[chunk["B"] >= 0.2, "recovery_rate_sig"].mean())
    fpr = float(chunk.loc[chunk["B"] == 0.0, "false_positive_rate_sig"].mean())
    return tpr, fpr


def _plot_tradeoff(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {BASELINE_VARIANT: "#1f77b4", ITEM2_VARIANT: "#d62728"}

    for _, row in comparison_df.iterrows():
        variant = str(row["detector_variant"])
        scenario = str(row["scenario"])
        if scenario == "target_group":
            marker = "D"
            size = 90
        else:
            marker = "o"
            size = 55
        ax.scatter(
            row["fpr_sig"],
            row["tpr_sig"],
            color=colors.get(variant, "black"),
            marker=marker,
            s=size,
            alpha=0.85,
        )
        ax.text(
            float(row["fpr_sig"]) + 0.002,
            float(row["tpr_sig"]) + 0.002,
            f"{scenario}:{variant.replace('_sig', '')}",
            fontsize=7,
        )

    ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.6, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR (B=0)")
    ax.set_ylabel("TPR (B>=0.2)")
    ax.set_title("Sprint 4 Item 2: Windowed FFT vs Welch FFT")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
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


def _write_log(
    log_path: Path,
    comparison_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_table = _markdown_table(comparison_df)
    decision_table = _markdown_table(decision_df)
    decision = bool(decision_df["improves_target_score"].iloc[0]) if not decision_df.empty else False
    next_step = "Keep Welch and proceed to broader validation." if decision else "No score gain; move to next detector family."
    text = f"""# Sprint 4 Item 2 Log (Welch Compare)

## Comparison Summary
{comparison_table}

## Decision
{decision_table}

## Recommended Next Step
{next_step}
"""
    log_path.write_text(text, encoding="utf-8")


def run_sprint4_welch_compare(
    config: BenchmarkConfig,
    output_dir: Path,
    figures_dir: Path,
    log_path: Path,
    scenario_map: Mapping[str, object] | None = None,
    max_points_for_bb: int = 2000,
    max_points_for_sig: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_map is None:
        scenario_map = get_default_scenarios()

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, pd.DataFrame] = {}
    for variant in (BASELINE_VARIANT, ITEM2_VARIANT):
        cfg = replace(config, detector_variant=variant)
        run_df, summary_df = run_rigor_benchmark(
            config=cfg,
            output_dir=output_dir / variant,
            figures_dir=figures_dir / "_sprint4_item2_tmp" / variant,
            paper_path=output_dir / "_scratch_paper.md",
            scenario_map=scenario_map,
            max_points_for_bb=max_points_for_bb,
            max_points_for_sig=max_points_for_sig,
            update_paper=False,
        )
        run_df.to_csv(output_dir / f"run_level_{variant}.csv", index=False)
        summary_df.to_csv(output_dir / f"summary_{variant}.csv", index=False)
        summaries[variant] = summary_df

    scenarios = sorted(set(config.scenario_names))
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        b_tpr, b_fpr = _scenario_metrics(summaries[BASELINE_VARIANT], scenario)
        w_tpr, w_fpr = _scenario_metrics(summaries[ITEM2_VARIANT], scenario)
        rows.append(
            {
                "scenario": scenario,
                "detector_variant": BASELINE_VARIANT,
                "tpr_sig": b_tpr,
                "fpr_sig": b_fpr,
                "score_tpr_minus_fpr": b_tpr - b_fpr if np.isfinite(b_tpr) and np.isfinite(b_fpr) else np.nan,
                "delta_score_vs_windowed": 0.0,
                "delta_tpr_vs_windowed": 0.0,
                "delta_fpr_vs_windowed": 0.0,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "detector_variant": ITEM2_VARIANT,
                "tpr_sig": w_tpr,
                "fpr_sig": w_fpr,
                "score_tpr_minus_fpr": w_tpr - w_fpr if np.isfinite(w_tpr) and np.isfinite(w_fpr) else np.nan,
                "delta_score_vs_windowed": (w_tpr - w_fpr) - (b_tpr - b_fpr)
                if np.isfinite(w_tpr) and np.isfinite(w_fpr) and np.isfinite(b_tpr) and np.isfinite(b_fpr)
                else np.nan,
                "delta_tpr_vs_windowed": w_tpr - b_tpr if np.isfinite(w_tpr) and np.isfinite(b_tpr) else np.nan,
                "delta_fpr_vs_windowed": w_fpr - b_fpr if np.isfinite(w_fpr) and np.isfinite(b_fpr) else np.nan,
            }
        )

    compare_df = pd.DataFrame(rows)
    base_target = compare_df[
        (compare_df["scenario"].isin(TARGET_SCENARIOS))
        & (compare_df["detector_variant"] == BASELINE_VARIANT)
    ]
    welch_target = compare_df[
        (compare_df["scenario"].isin(TARGET_SCENARIOS))
        & (compare_df["detector_variant"] == ITEM2_VARIANT)
    ]
    base_score = float(base_target["score_tpr_minus_fpr"].mean())
    welch_score = float(welch_target["score_tpr_minus_fpr"].mean())
    base_tpr = float(base_target["tpr_sig"].mean())
    welch_tpr = float(welch_target["tpr_sig"].mean())
    base_fpr = float(base_target["fpr_sig"].mean())
    welch_fpr = float(welch_target["fpr_sig"].mean())

    compare_df = pd.concat(
        [
            compare_df,
            pd.DataFrame(
                [
                    {
                        "scenario": "target_group",
                        "detector_variant": BASELINE_VARIANT,
                        "tpr_sig": base_tpr,
                        "fpr_sig": base_fpr,
                        "score_tpr_minus_fpr": base_score,
                        "delta_score_vs_windowed": 0.0,
                        "delta_tpr_vs_windowed": 0.0,
                        "delta_fpr_vs_windowed": 0.0,
                    },
                    {
                        "scenario": "target_group",
                        "detector_variant": ITEM2_VARIANT,
                        "tpr_sig": welch_tpr,
                        "fpr_sig": welch_fpr,
                        "score_tpr_minus_fpr": welch_score,
                        "delta_score_vs_windowed": welch_score - base_score,
                        "delta_tpr_vs_windowed": welch_tpr - base_tpr,
                        "delta_fpr_vs_windowed": welch_fpr - base_fpr,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    decision_df = pd.DataFrame(
        [
            {
                "baseline_variant": BASELINE_VARIANT,
                "item2_variant": ITEM2_VARIANT,
                "target_score_windowed": base_score,
                "target_score_welch": welch_score,
                "delta_score": welch_score - base_score,
                "delta_tpr": welch_tpr - base_tpr,
                "delta_fpr": welch_fpr - base_fpr,
                "improves_target_score": bool((welch_score - base_score) > 0.0),
            }
        ]
    )

    compare_csv = output_dir / "welch_comparison.csv"
    decision_csv = output_dir / "welch_decision.csv"
    compare_df.to_csv(compare_csv, index=False)
    decision_df.to_csv(decision_csv, index=False)

    tradeoff_png = figures_dir / "sprint4_welch_tradeoff.png"
    _plot_tradeoff(compare_df, tradeoff_png)
    _write_log(log_path, comparison_df=compare_df, decision_df=decision_df)

    print("[sprint4-welch] complete")
    print(f"[sprint4-welch] comparison: {compare_csv}")
    print(f"[sprint4-welch] decision: {decision_csv}")
    print(f"[sprint4-welch] figure: {tradeoff_png}")
    print(f"[sprint4-welch] log: {log_path}")
    return compare_df, decision_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 4 Item 2: compare Welch and windowed FFT significance detectors.")
    parser.add_argument("--output-dir", default="outputs/sprint4_welch_compare")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--log-path", default="docs/sprint4_welch_log.md")
    parser.add_argument("--scenarios", default="short_70,short_100,weak,mid,boat_drift,boat_transient")
    parser.add_argument("--b-grid", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--p0-scales", default="0.5,1.0,2.0")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.02)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--freq-band-min", type=float, default=0.35)
    parser.add_argument("--freq-band-max", type=float, default=0.45)
    parser.add_argument("--window-padding-s", type=float, default=10.0)
    parser.add_argument("--welch-segment-points", type=int, default=256)
    parser.add_argument("--welch-overlap-frac", type=float, default=0.5)
    parser.add_argument("--max-points-for-bb", type=int, default=2000)
    parser.add_argument("--max-points-for-sig", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BenchmarkConfig(
        scenario_names=tuple(_parse_str_list(args.scenarios)),
        b_grid=tuple(_parse_float_list(args.b_grid)),
        p0_scale_grid=tuple(_parse_float_list(args.p0_scales)),
        n_replicates=int(args.n_replicates),
        freq_tolerance_hz=float(args.freq_tolerance_hz),
        seed_start=int(args.seed_start),
        n_surrogates=int(args.n_surrogates),
        alpha=float(args.alpha),
        freq_band_min=float(args.freq_band_min),
        freq_band_max=float(args.freq_band_max),
        window_padding_s=float(args.window_padding_s),
        welch_segment_points=int(args.welch_segment_points),
        welch_overlap_frac=float(args.welch_overlap_frac),
    )
    run_sprint4_welch_compare(
        config=cfg,
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        log_path=Path(args.log_path),
        max_points_for_bb=int(args.max_points_for_bb),
        max_points_for_sig=int(args.max_points_for_sig),
    )

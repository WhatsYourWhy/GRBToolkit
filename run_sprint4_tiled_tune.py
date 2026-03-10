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

BASELINE_VARIANT = "windowed_fft_sig"
TILED_VARIANT = "tiled_window_fft_sig"
TARGET_SCENARIOS = ("mid", "boat_drift")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _target_metrics(summary_df: pd.DataFrame, target_scenarios: Sequence[str]) -> tuple[float, float, float]:
    chunk = summary_df[summary_df["scenario"].isin(target_scenarios)]
    if chunk.empty:
        return float("nan"), float("nan"), float("nan")
    tpr = float(chunk.loc[chunk["B"] >= 0.2, "recovery_rate_sig"].mean())
    fpr = float(chunk.loc[chunk["B"] == 0.0, "false_positive_rate_sig"].mean())
    score = float(tpr - fpr) if np.isfinite(tpr) and np.isfinite(fpr) else float("nan")
    return tpr, fpr, score


def _format_markdown_table(df: pd.DataFrame) -> str:
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


def _plot_tune_scatter(candidate_df: pd.DataFrame, output_path: Path) -> None:
    if candidate_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    markers = {"bh": "o", "bonferroni": "s"}

    for _, row in candidate_df.iterrows():
        marker = markers.get(str(row["tile_correction_method"]), "o")
        ax.scatter(row["delta_fpr"], row["delta_tpr"], marker=marker, s=70, alpha=0.85)
        ax.text(
            float(row["delta_fpr"]) + 0.001,
            float(row["delta_tpr"]) + 0.001,
            str(row["candidate_id"]),
            fontsize=7,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Delta FPR vs baseline (negative is better)")
    ax.set_ylabel("Delta TPR vs baseline")
    ax.set_title("Sprint 4 Item 3 Tune: Candidate Delta Map")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_decision_table(
    candidate_df: pd.DataFrame,
    min_delta_tpr: float,
    max_fpr_target: float,
    min_delta_score: float,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(
            [
                {
                    "best_candidate_id": "none",
                "best_delta_tpr": np.nan,
                "best_delta_fpr": np.nan,
                "best_delta_score": np.nan,
                "delta_score": np.nan,
                "best_fpr_target": np.nan,
                "best_tpr_target": np.nan,
                    "meets_delta_tpr_bar": False,
                    "meets_fpr_bar": False,
                    "meets_delta_score_bar": False,
                    "recommended_action": "STOP_OR_PIVOT",
                    "reason": "No candidates evaluated",
                }
            ]
        )

    best = candidate_df.iloc[0]
    meets_delta_tpr = bool(float(best["delta_tpr"]) >= float(min_delta_tpr))
    meets_fpr = bool(float(best["fpr_sig_target"]) <= float(max_fpr_target))
    meets_delta_score = bool(float(best["delta_score"]) > float(min_delta_score))

    if meets_delta_tpr and meets_fpr and meets_delta_score:
        action = "GO_KEEP_TILED"
        reason = "Best candidate improves TPR with controlled FPR and positive score gain."
    else:
        max_delta_tpr = float(candidate_df["delta_tpr"].max())
        if max_delta_tpr < float(min_delta_tpr):
            action = "STOP_OR_PIVOT"
            reason = "No candidate reached minimum TPR gain threshold."
        else:
            action = "TUNE_MORE"
            reason = "Some TPR movement exists, but score/FPR gate is not fully met."

    return pd.DataFrame(
        [
            {
                "best_candidate_id": best["candidate_id"],
                "best_delta_tpr": float(best["delta_tpr"]),
                "best_delta_fpr": float(best["delta_fpr"]),
                "best_delta_score": float(best["delta_score"]),
                "delta_score": float(best["delta_score"]),
                "best_fpr_target": float(best["fpr_sig_target"]),
                "best_tpr_target": float(best["tpr_sig_target"]),
                "meets_delta_tpr_bar": meets_delta_tpr,
                "meets_fpr_bar": meets_fpr,
                "meets_delta_score_bar": meets_delta_score,
                "recommended_action": action,
                "reason": reason,
            }
        ]
    )


def _write_log(
    log_path: Path,
    baseline_metrics: dict[str, float],
    candidate_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    top = candidate_df.head(10) if not candidate_df.empty else candidate_df
    top_table = _format_markdown_table(top) if not top.empty else "_No candidate rows._"
    decision_table = _format_markdown_table(decision_df)

    text = f"""# Sprint 4 Item 3 Tuning Log

## Baseline (Windowed FFT)
- Target TPR: `{baseline_metrics['tpr']:.4f}`
- Target FPR: `{baseline_metrics['fpr']:.4f}`
- Target score (`TPR-FPR`): `{baseline_metrics['score']:.4f}`

## Candidate Ranking (Top 10)
{top_table}

## Stop/Go Decision
{decision_table}
"""
    log_path.write_text(text, encoding="utf-8")


def run_sprint4_tiled_tune(
    config: BenchmarkConfig,
    output_dir: Path,
    figures_dir: Path,
    log_path: Path,
    tile_window_grid_s: Sequence[float],
    tile_step_grid_s: Sequence[float],
    tile_correction_methods: Sequence[str],
    tile_max_windows_grid: Sequence[int],
    min_delta_tpr: float = 0.05,
    max_fpr_target: float = 0.1,
    min_delta_score: float = 0.0,
    scenario_map: Mapping[str, object] | None = None,
    max_points_for_bb: int = 2000,
    max_points_for_sig: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_map is None:
        scenario_map = get_default_scenarios()

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_cfg = replace(config, detector_variant=BASELINE_VARIANT)
    base_run, base_summary = run_rigor_benchmark(
        config=baseline_cfg,
        output_dir=output_dir / BASELINE_VARIANT,
        figures_dir=figures_dir / "_sprint4_item3_tune_tmp" / BASELINE_VARIANT,
        paper_path=output_dir / "_scratch_paper.md",
        scenario_map=scenario_map,
        max_points_for_bb=max_points_for_bb,
        max_points_for_sig=max_points_for_sig,
        update_paper=False,
    )
    base_run.to_csv(output_dir / f"run_level_{BASELINE_VARIANT}.csv", index=False)
    base_summary.to_csv(output_dir / f"summary_{BASELINE_VARIANT}.csv", index=False)
    base_tpr, base_fpr, base_score = _target_metrics(base_summary, TARGET_SCENARIOS)

    rows: list[dict[str, object]] = []
    candidate_idx = 0
    for tile_window_s in tile_window_grid_s:
        for tile_step_s in tile_step_grid_s:
            if float(tile_step_s) > float(tile_window_s):
                continue
            for correction in tile_correction_methods:
                for tile_max_windows in tile_max_windows_grid:
                    candidate_id = f"T{candidate_idx:02d}"
                    candidate_idx += 1
                    cfg = replace(
                        config,
                        detector_variant=TILED_VARIANT,
                        tile_window_s=float(tile_window_s),
                        tile_step_s=float(tile_step_s),
                        tile_correction_method=str(correction),
                        tile_max_windows=int(tile_max_windows),
                    )
                    run_df, summary_df = run_rigor_benchmark(
                        config=cfg,
                        output_dir=output_dir / "candidates" / candidate_id,
                        figures_dir=figures_dir / "_sprint4_item3_tune_tmp" / candidate_id,
                        paper_path=output_dir / "_scratch_paper.md",
                        scenario_map=scenario_map,
                        max_points_for_bb=max_points_for_bb,
                        max_points_for_sig=max_points_for_sig,
                        update_paper=False,
                    )
                    run_df.to_csv(output_dir / f"run_level_{candidate_id}.csv", index=False)
                    summary_df.to_csv(output_dir / f"summary_{candidate_id}.csv", index=False)

                    tpr, fpr, score = _target_metrics(summary_df, TARGET_SCENARIOS)
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "tile_window_s": float(tile_window_s),
                            "tile_step_s": float(tile_step_s),
                            "tile_correction_method": str(correction),
                            "tile_max_windows": int(tile_max_windows),
                            "tpr_sig_target": tpr,
                            "fpr_sig_target": fpr,
                            "score_tpr_minus_fpr_target": score,
                            "delta_tpr": float(tpr - base_tpr) if np.isfinite(tpr) and np.isfinite(base_tpr) else np.nan,
                            "delta_fpr": float(fpr - base_fpr) if np.isfinite(fpr) and np.isfinite(base_fpr) else np.nan,
                            "delta_score": float(score - base_score) if np.isfinite(score) and np.isfinite(base_score) else np.nan,
                            "passes_balanced_bar": bool(
                                np.isfinite(tpr)
                                and np.isfinite(fpr)
                                and (tpr >= 0.6)
                                and (fpr <= 0.1)
                            ),
                        }
                    )

    candidate_df = pd.DataFrame(rows)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(
            ["delta_score", "delta_tpr", "delta_fpr"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        candidate_df["rank"] = np.arange(1, len(candidate_df) + 1, dtype=int)

    decision_df = _build_decision_table(
        candidate_df=candidate_df,
        min_delta_tpr=min_delta_tpr,
        max_fpr_target=max_fpr_target,
        min_delta_score=min_delta_score,
    )

    candidate_csv = output_dir / "tiled_tune_candidates.csv"
    best_csv = output_dir / "tiled_tune_best.csv"
    decision_csv = output_dir / "tiled_tune_decision.csv"
    candidate_df.to_csv(candidate_csv, index=False)
    candidate_df.head(1).to_csv(best_csv, index=False)
    decision_df.to_csv(decision_csv, index=False)

    scatter_png = figures_dir / "sprint4_tiled_tune_delta_scatter.png"
    _plot_tune_scatter(candidate_df, scatter_png)

    _write_log(
        log_path=log_path,
        baseline_metrics={"tpr": base_tpr, "fpr": base_fpr, "score": base_score},
        candidate_df=candidate_df,
        decision_df=decision_df,
    )

    print("[sprint4-tiled-tune] complete")
    print(f"[sprint4-tiled-tune] candidates: {candidate_csv}")
    print(f"[sprint4-tiled-tune] best: {best_csv}")
    print(f"[sprint4-tiled-tune] decision: {decision_csv}")
    print(f"[sprint4-tiled-tune] figure: {scatter_png}")
    print(f"[sprint4-tiled-tune] log: {log_path}")
    return candidate_df, decision_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 4 Item 3 tuning sweep for tiled-window significance detector."
    )
    parser.add_argument("--output-dir", default="docs/sprint4_tiled_tune")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--log-path", default="docs/sprint4_tiled_tune_log.md")
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
    parser.add_argument("--tile-window-grid-s", default="40,60,80")
    parser.add_argument("--tile-step-grid-s", default="10,20")
    parser.add_argument("--tile-correction-methods", default="bh,bonferroni")
    parser.add_argument("--tile-max-windows-grid", default="8,12")
    parser.add_argument("--tile-min-points", type=int, default=64)
    parser.add_argument("--min-delta-tpr", type=float, default=0.05)
    parser.add_argument("--max-fpr-target", type=float, default=0.1)
    parser.add_argument("--min-delta-score", type=float, default=0.0)
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
        tile_min_points=int(args.tile_min_points),
    )
    run_sprint4_tiled_tune(
        config=cfg,
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        log_path=Path(args.log_path),
        tile_window_grid_s=_parse_float_list(args.tile_window_grid_s),
        tile_step_grid_s=_parse_float_list(args.tile_step_grid_s),
        tile_correction_methods=tuple(_parse_str_list(args.tile_correction_methods)),
        tile_max_windows_grid=tuple(_parse_int_list(args.tile_max_windows_grid)),
        min_delta_tpr=float(args.min_delta_tpr),
        max_fpr_target=float(args.max_fpr_target),
        min_delta_score=float(args.min_delta_score),
        max_points_for_bb=int(args.max_points_for_bb),
        max_points_for_sig=int(args.max_points_for_sig),
    )

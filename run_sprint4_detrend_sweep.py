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
ITEM4_VARIANT = "detrended_fft_sig"
TARGET_SCENARIOS = ("mid", "boat_drift")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _scenario_metrics(summary_df: pd.DataFrame, scenario: str) -> tuple[float, float]:
    chunk = summary_df[summary_df["scenario"] == scenario]
    if chunk.empty:
        return float("nan"), float("nan")
    tpr = float(chunk.loc[chunk["B"] >= 0.2, "recovery_rate_sig"].mean())
    fpr = float(chunk.loc[chunk["B"] == 0.0, "false_positive_rate_sig"].mean())
    return tpr, fpr


def _target_metrics(summary_df: pd.DataFrame) -> tuple[float, float, float, bool]:
    per_scenario: list[tuple[float, float, bool]] = []
    for scenario in TARGET_SCENARIOS:
        tpr, fpr = _scenario_metrics(summary_df, scenario)
        passes = bool(np.isfinite(tpr) and np.isfinite(fpr) and (tpr >= 0.6) and (fpr <= 0.1))
        per_scenario.append((tpr, fpr, passes))

    tpr_mean = _safe_mean([row[0] for row in per_scenario])
    fpr_mean = _safe_mean([row[1] for row in per_scenario])
    score = float(tpr_mean - fpr_mean) if np.isfinite(tpr_mean) and np.isfinite(fpr_mean) else float("nan")
    passes_all = bool(all(row[2] for row in per_scenario))
    return tpr_mean, fpr_mean, score, passes_all


def _markdown_table(df: pd.DataFrame) -> str:
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


def _plot_tradeoff(candidate_df: pd.DataFrame, output_path: Path) -> None:
    if candidate_df.empty:
        return

    splits = [split for split in ("train", "holdout") if split in candidate_df["split"].unique().tolist()]
    if not splits:
        return

    fig, axes = plt.subplots(1, len(splits), figsize=(7.4 * len(splits), 5.0), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for idx, split in enumerate(splits):
        ax = axes[0][idx]
        subset = candidate_df[candidate_df["split"] == split].copy()
        base = subset[subset["is_baseline"]]
        candidates = subset[~subset["is_baseline"]]

        if not base.empty:
            row = base.iloc[0]
            ax.scatter(
                row["fpr_sig_target"],
                row["tpr_sig_target"],
                marker="*",
                s=180,
                color="black",
                label="baseline",
            )

        for c_idx, (_, row) in enumerate(candidates.iterrows()):
            color = cmap(c_idx % 10)
            ax.scatter(row["fpr_sig_target"], row["tpr_sig_target"], s=62, color=color, alpha=0.85)
            ax.text(
                float(row["fpr_sig_target"]) + 0.002,
                float(row["tpr_sig_target"]) + 0.002,
                f"{row['candidate_id']}(o{int(row['detrend_order'])})",
                fontsize=7,
            )

        ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(0.6, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Target-group FPR mean")
        ax.set_ylabel("Target-group TPR mean")
        ax.set_title(f"Sprint 4 Item 4 ({split})")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_delta(candidate_df: pd.DataFrame, output_path: Path) -> None:
    holdout = candidate_df[(candidate_df["split"] == "holdout") & (~candidate_df["is_baseline"])].copy()
    if holdout.empty:
        return

    holdout = holdout.sort_values(["delta_score", "delta_tpr"], ascending=[False, False]).reset_index(drop=True)
    x = np.arange(len(holdout))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(x, holdout["delta_tpr"], marker="o", label="delta_tpr")
    ax.plot(x, holdout["delta_fpr"], marker="s", label="delta_fpr")
    ax.plot(x, holdout["delta_score"], marker="^", label="delta_score")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cid}(o{int(order)})" for cid, order in zip(holdout["candidate_id"], holdout["detrend_order"])], rotation=35, ha="right")
    ax.set_ylabel("Delta vs windowed baseline")
    ax.set_title("Sprint 4 Item 4: Holdout Delta by Detrend Order")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_decision(
    candidate_df: pd.DataFrame,
    min_delta_tpr: float,
    max_fpr_target: float,
    min_delta_score: float,
) -> pd.DataFrame:
    holdout = candidate_df[(candidate_df["split"] == "holdout") & (~candidate_df["is_baseline"])].copy()
    train = candidate_df[(candidate_df["split"] == "train") & (~candidate_df["is_baseline"])].copy()
    if holdout.empty:
        return pd.DataFrame(
            [
                {
                    "best_candidate_id": "none",
                    "best_detrend_order": np.nan,
                    "holdout_delta_tpr": np.nan,
                    "holdout_delta_fpr": np.nan,
                    "holdout_delta_score": np.nan,
                    "holdout_tpr_target": np.nan,
                    "holdout_fpr_target": np.nan,
                    "train_delta_tpr": np.nan,
                    "train_delta_fpr": np.nan,
                    "train_delta_score": np.nan,
                    "meets_delta_tpr_bar": False,
                    "meets_fpr_bar": False,
                    "meets_delta_score_bar": False,
                    "meets_train_consistency": False,
                    "recommended_action": "STOP_OR_PIVOT",
                    "reason": "No detrend candidates were evaluated on holdout seeds.",
                }
            ]
        )

    holdout = holdout.sort_values(["delta_score", "delta_tpr", "delta_fpr"], ascending=[False, False, True]).reset_index(drop=True)
    best = holdout.iloc[0]

    train_match = train[train["candidate_id"] == best["candidate_id"]]
    if train_match.empty:
        train_delta_tpr = float("nan")
        train_delta_fpr = float("nan")
        train_delta_score = float("nan")
        train_consistent = False
    else:
        train_row = train_match.iloc[0]
        train_delta_tpr = float(train_row["delta_tpr"])
        train_delta_fpr = float(train_row["delta_fpr"])
        train_delta_score = float(train_row["delta_score"])
        train_consistent = bool(np.isfinite(train_delta_tpr) and np.isfinite(train_delta_score) and (train_delta_tpr >= 0.0) and (train_delta_score >= 0.0))

    meets_delta_tpr = bool(float(best["delta_tpr"]) >= float(min_delta_tpr))
    meets_fpr = bool(float(best["fpr_sig_target"]) <= float(max_fpr_target))
    meets_delta_score = bool(float(best["delta_score"]) > float(min_delta_score))

    if meets_delta_tpr and meets_fpr and meets_delta_score and train_consistent:
        action = "GO_KEEP_DETREND"
        reason = "Best holdout candidate improves TPR with controlled FPR and matches non-negative train deltas."
    else:
        max_delta_tpr = float(holdout["delta_tpr"].max())
        if max_delta_tpr < float(min_delta_tpr):
            action = "STOP_OR_PIVOT"
            reason = "No holdout candidate reached minimum target-group TPR gain threshold."
        else:
            action = "TUNE_MORE"
            reason = "Some TPR movement exists but score/FPR/train-consistency gate is not met."

    return pd.DataFrame(
        [
            {
                "best_candidate_id": best["candidate_id"],
                "best_detrend_order": int(best["detrend_order"]),
                "holdout_delta_tpr": float(best["delta_tpr"]),
                "holdout_delta_fpr": float(best["delta_fpr"]),
                "holdout_delta_score": float(best["delta_score"]),
                "holdout_tpr_target": float(best["tpr_sig_target"]),
                "holdout_fpr_target": float(best["fpr_sig_target"]),
                "train_delta_tpr": train_delta_tpr,
                "train_delta_fpr": train_delta_fpr,
                "train_delta_score": train_delta_score,
                "meets_delta_tpr_bar": meets_delta_tpr,
                "meets_fpr_bar": meets_fpr,
                "meets_delta_score_bar": meets_delta_score,
                "meets_train_consistency": train_consistent,
                "recommended_action": action,
                "reason": reason,
            }
        ]
    )


def _write_log(
    log_path: Path,
    baseline_table: pd.DataFrame,
    holdout_top: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_md = _markdown_table(baseline_table)
    holdout_md = _markdown_table(holdout_top)
    decision_md = _markdown_table(decision_df)

    text = f"""# Sprint 4 Item 4 Log (Detrend Sweep)

## Baseline Summary
{baseline_md}

## Holdout Candidate Ranking (Top 10)
{holdout_md}

## Decision
{decision_md}
"""
    log_path.write_text(text, encoding="utf-8")


def run_sprint4_detrend_sweep(
    config: BenchmarkConfig,
    output_dir: Path,
    figures_dir: Path,
    log_path: Path,
    detrend_order_grid: Sequence[int],
    holdout_seed_offset: int = 5_000_000,
    min_delta_tpr: float = 0.05,
    max_fpr_target: float = 0.1,
    min_delta_score: float = 0.0,
    scenario_map: Mapping[str, object] | None = None,
    max_points_for_bb: int = 2000,
    max_points_for_sig: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_map is None:
        scenario_map = get_default_scenarios()

    if not detrend_order_grid:
        raise ValueError("detrend_order_grid cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    split_seed_start = {
        "train": int(config.seed_start),
        "holdout": int(config.seed_start) + int(holdout_seed_offset),
    }

    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    for split, seed_start in split_seed_start.items():
        baseline_run_csv = output_dir / f"run_level_{split}_baseline.csv"
        baseline_summary_csv = output_dir / f"summary_{split}_baseline.csv"
        baseline_cfg = replace(
            config,
            detector_variant=BASELINE_VARIANT,
            seed_start=int(seed_start),
        )
        if baseline_run_csv.exists() and baseline_summary_csv.exists():
            base_run = pd.read_csv(baseline_run_csv)
            base_summary = pd.read_csv(baseline_summary_csv)
            print(f"[sprint4-detrend] resume: using existing baseline files for split='{split}'")
        else:
            base_run, base_summary = run_rigor_benchmark(
                config=baseline_cfg,
                output_dir=output_dir / split / BASELINE_VARIANT,
                figures_dir=figures_dir / "_sprint4_item4_tmp" / split / BASELINE_VARIANT,
                paper_path=output_dir / "_scratch_paper.md",
                scenario_map=scenario_map,
                max_points_for_bb=max_points_for_bb,
                max_points_for_sig=max_points_for_sig,
                update_paper=False,
            )
            base_run.to_csv(baseline_run_csv, index=False)
            base_summary.to_csv(baseline_summary_csv, index=False)

        base_tpr, base_fpr, base_score, base_pass = _target_metrics(base_summary)
        baseline_rows.append(
            {
                "split": split,
                "tpr_sig_target": base_tpr,
                "fpr_sig_target": base_fpr,
                "score_tpr_minus_fpr_target": base_score,
                "passes_balanced_bar": base_pass,
            }
        )
        rows.append(
            {
                "split": split,
                "candidate_id": "BASE",
                "detector_variant": BASELINE_VARIANT,
                "detrend_order": 0,
                "tpr_sig_target": base_tpr,
                "fpr_sig_target": base_fpr,
                "score_tpr_minus_fpr_target": base_score,
                "delta_tpr": 0.0,
                "delta_fpr": 0.0,
                "delta_score": 0.0,
                "passes_balanced_bar": bool(base_pass),
                "is_baseline": True,
            }
        )

        for idx, detrend_order in enumerate(detrend_order_grid):
            candidate_id = f"D{idx:02d}"
            candidate_run_csv = output_dir / f"run_level_{split}_{candidate_id}.csv"
            candidate_summary_csv = output_dir / f"summary_{split}_{candidate_id}.csv"
            candidate_cfg = replace(
                config,
                detector_variant=ITEM4_VARIANT,
                detrend_order=int(detrend_order),
                seed_start=int(seed_start),
            )
            if candidate_run_csv.exists() and candidate_summary_csv.exists():
                run_df = pd.read_csv(candidate_run_csv)
                summary_df = pd.read_csv(candidate_summary_csv)
                print(f"[sprint4-detrend] resume: using existing candidate files for split='{split}', {candidate_id}")
            else:
                run_df, summary_df = run_rigor_benchmark(
                    config=candidate_cfg,
                    output_dir=output_dir / split / candidate_id,
                    figures_dir=figures_dir / "_sprint4_item4_tmp" / split / candidate_id,
                    paper_path=output_dir / "_scratch_paper.md",
                    scenario_map=scenario_map,
                    max_points_for_bb=max_points_for_bb,
                    max_points_for_sig=max_points_for_sig,
                    update_paper=False,
                )
                run_df.to_csv(candidate_run_csv, index=False)
                summary_df.to_csv(candidate_summary_csv, index=False)

            tpr, fpr, score, passes = _target_metrics(summary_df)
            rows.append(
                {
                    "split": split,
                    "candidate_id": candidate_id,
                    "detector_variant": ITEM4_VARIANT,
                    "detrend_order": int(detrend_order),
                    "tpr_sig_target": tpr,
                    "fpr_sig_target": fpr,
                    "score_tpr_minus_fpr_target": score,
                    "delta_tpr": float(tpr - base_tpr) if np.isfinite(tpr) and np.isfinite(base_tpr) else np.nan,
                    "delta_fpr": float(fpr - base_fpr) if np.isfinite(fpr) and np.isfinite(base_fpr) else np.nan,
                    "delta_score": float(score - base_score) if np.isfinite(score) and np.isfinite(base_score) else np.nan,
                    "passes_balanced_bar": bool(passes),
                    "is_baseline": False,
                }
            )

    candidate_df = pd.DataFrame(rows).sort_values(
        ["split", "is_baseline", "delta_score", "delta_tpr", "delta_fpr"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    decision_df = _build_decision(
        candidate_df=candidate_df,
        min_delta_tpr=min_delta_tpr,
        max_fpr_target=max_fpr_target,
        min_delta_score=min_delta_score,
    )

    holdout_top = candidate_df[
        (candidate_df["split"] == "holdout") & (~candidate_df["is_baseline"])
    ].sort_values(["delta_score", "delta_tpr", "delta_fpr"], ascending=[False, False, True]).head(10)

    baseline_table = pd.DataFrame(baseline_rows)
    candidate_csv = output_dir / "detrend_candidate_summary.csv"
    best_csv = output_dir / "detrend_best_candidate.csv"
    decision_csv = output_dir / "detrend_decision.csv"
    candidate_df.to_csv(candidate_csv, index=False)
    holdout_top.head(1).to_csv(best_csv, index=False)
    decision_df.to_csv(decision_csv, index=False)

    tradeoff_png = figures_dir / "sprint4_detrend_tradeoff.png"
    delta_png = figures_dir / "sprint4_detrend_delta.png"
    _plot_tradeoff(candidate_df, tradeoff_png)
    _plot_delta(candidate_df, delta_png)
    _write_log(log_path, baseline_table=baseline_table, holdout_top=holdout_top, decision_df=decision_df)

    print("[sprint4-detrend] complete")
    print(f"[sprint4-detrend] candidate summary: {candidate_csv}")
    print(f"[sprint4-detrend] best candidate: {best_csv}")
    print(f"[sprint4-detrend] decision: {decision_csv}")
    print(f"[sprint4-detrend] figures: {tradeoff_png}, {delta_png}")
    print(f"[sprint4-detrend] log: {log_path}")
    return candidate_df, decision_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 4 Item 4: detrend-order sweep with holdout-seed decision gating."
    )
    parser.add_argument("--output-dir", default="outputs/sprint4_detrend_sweep")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--log-path", default="docs/sprint4_detrend_log.md")
    parser.add_argument("--scenarios", default="short_70,short_100,weak,mid,boat_drift,boat_transient")
    parser.add_argument("--b-grid", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--p0-scales", default="0.5,1.0,2.0")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.02)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--holdout-seed-offset", type=int, default=5_000_000)
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--freq-band-min", type=float, default=0.35)
    parser.add_argument("--freq-band-max", type=float, default=0.45)
    parser.add_argument("--window-padding-s", type=float, default=10.0)
    parser.add_argument("--detrend-order-grid", default="1,2,3")
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
    )
    run_sprint4_detrend_sweep(
        config=cfg,
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        log_path=Path(args.log_path),
        detrend_order_grid=tuple(_parse_int_list(args.detrend_order_grid)),
        holdout_seed_offset=int(args.holdout_seed_offset),
        min_delta_tpr=float(args.min_delta_tpr),
        max_fpr_target=float(args.max_fpr_target),
        min_delta_score=float(args.min_delta_score),
        max_points_for_bb=int(args.max_points_for_bb),
        max_points_for_sig=int(args.max_points_for_sig),
    )

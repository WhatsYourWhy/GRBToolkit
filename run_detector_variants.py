from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grb_refresh import BenchmarkConfig, get_default_scenarios
from run_rigor_benchmark import VALID_DETECTOR_VARIANTS, run_rigor_benchmark

BASELINE_VARIANT = "global_tapered_fft_sig"
TARGET_SCENARIOS = ("mid", "boat_drift")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_variants(variants: Sequence[str]) -> tuple[str, ...]:
    unique = []
    for variant in variants:
        if variant not in VALID_DETECTOR_VARIANTS:
            allowed = ", ".join(VALID_DETECTOR_VARIANTS)
            raise ValueError(f"Unknown detector variant '{variant}'. Allowed: {allowed}")
        if variant not in unique:
            unique.append(variant)

    if BASELINE_VARIANT not in unique:
        unique.insert(0, BASELINE_VARIANT)
    return tuple(unique)


def _metric_for_variant(summary_df: pd.DataFrame, scenario: str) -> tuple[float, float]:
    scenario_df = summary_df[summary_df["scenario"] == scenario]
    if scenario_df.empty:
        return float("nan"), float("nan")

    fpr = float(scenario_df.loc[scenario_df["B"] == 0.0, "false_positive_rate_sig"].mean())
    tpr = float(scenario_df.loc[scenario_df["B"] >= 0.2, "recovery_rate_sig"].mean())
    return tpr, fpr


def build_variant_comparison(summary_by_variant: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    scenarios = sorted(
        {
            scenario
            for df in summary_by_variant.values()
            for scenario in df["scenario"].unique().tolist()
        }
    )
    baseline_df = summary_by_variant.get(BASELINE_VARIANT)
    if baseline_df is None:
        raise ValueError(f"Baseline variant '{BASELINE_VARIANT}' missing from summary_by_variant")

    baseline_metrics = {scenario: _metric_for_variant(baseline_df, scenario) for scenario in scenarios}

    rows: list[dict[str, object]] = []
    for variant, summary_df in summary_by_variant.items():
        for scenario in scenarios:
            tpr_sig, fpr_sig = _metric_for_variant(summary_df, scenario)
            baseline_tpr, baseline_fpr = baseline_metrics.get(scenario, (float("nan"), float("nan")))

            delta_tpr = float(tpr_sig - baseline_tpr) if np.isfinite(tpr_sig) and np.isfinite(baseline_tpr) else np.nan
            delta_fpr = float(fpr_sig - baseline_fpr) if np.isfinite(fpr_sig) and np.isfinite(baseline_fpr) else np.nan
            passes_balanced_bar = bool(
                scenario in TARGET_SCENARIOS
                and np.isfinite(tpr_sig)
                and np.isfinite(fpr_sig)
                and (tpr_sig >= 0.6)
                and (fpr_sig <= 0.1)
            )

            rows.append(
                {
                    "scenario": scenario,
                    "detector_variant": variant,
                    "B_bucket": "B0",
                    "TPR_sig": np.nan,
                    "FPR_sig": fpr_sig,
                    "delta_tpr_vs_baseline": np.nan,
                    "delta_fpr_vs_baseline": delta_fpr,
                    "passes_balanced_bar": passes_balanced_bar,
                }
            )
            rows.append(
                {
                    "scenario": scenario,
                    "detector_variant": variant,
                    "B_bucket": "B>=0.2",
                    "TPR_sig": tpr_sig,
                    "FPR_sig": fpr_sig,
                    "delta_tpr_vs_baseline": delta_tpr,
                    "delta_fpr_vs_baseline": delta_fpr,
                    "passes_balanced_bar": passes_balanced_bar,
                }
            )

    return pd.DataFrame(rows)[
        [
            "scenario",
            "detector_variant",
            "B_bucket",
            "TPR_sig",
            "FPR_sig",
            "delta_tpr_vs_baseline",
            "delta_fpr_vs_baseline",
            "passes_balanced_bar",
        ]
    ].sort_values(["scenario", "detector_variant", "B_bucket"]).reset_index(drop=True)


def build_best_variant_selector(comparison_df: pd.DataFrame) -> pd.DataFrame:
    subset = comparison_df[comparison_df["B_bucket"] == "B>=0.2"].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "selector_scope",
                "selector_key",
                "best_detector_variant",
                "TPR_sig",
                "FPR_sig",
                "score_tpr_minus_fpr",
                "passes_balanced_bar",
            ]
        )

    subset["score_tpr_minus_fpr"] = subset["TPR_sig"] - subset["FPR_sig"]
    rows: list[dict[str, object]] = []

    for scenario, group in subset.groupby("scenario"):
        ordered = group.sort_values(
            ["score_tpr_minus_fpr", "TPR_sig", "FPR_sig"],
            ascending=[False, False, True],
        )
        best = ordered.iloc[0]
        rows.append(
            {
                "selector_scope": "scenario",
                "selector_key": scenario,
                "best_detector_variant": str(best["detector_variant"]),
                "TPR_sig": float(best["TPR_sig"]),
                "FPR_sig": float(best["FPR_sig"]),
                "score_tpr_minus_fpr": float(best["score_tpr_minus_fpr"]),
                "passes_balanced_bar": bool(best["passes_balanced_bar"]),
            }
        )

    target_group = subset[subset["scenario"].isin(TARGET_SCENARIOS)].copy()
    if not target_group.empty:
        grouped = (
            target_group.groupby("detector_variant", as_index=False)
            .agg(
                TPR_sig=("TPR_sig", "mean"),
                FPR_sig=("FPR_sig", "mean"),
                passes_balanced_bar=("passes_balanced_bar", "all"),
            )
            .copy()
        )
        grouped["score_tpr_minus_fpr"] = grouped["TPR_sig"] - grouped["FPR_sig"]
        grouped = grouped.sort_values(
            ["score_tpr_minus_fpr", "TPR_sig", "FPR_sig"],
            ascending=[False, False, True],
        )
        best = grouped.iloc[0]
        rows.append(
            {
                "selector_scope": "target_group",
                "selector_key": "+".join(TARGET_SCENARIOS),
                "best_detector_variant": str(best["detector_variant"]),
                "TPR_sig": float(best["TPR_sig"]),
                "FPR_sig": float(best["FPR_sig"]),
                "score_tpr_minus_fpr": float(best["score_tpr_minus_fpr"]),
                "passes_balanced_bar": bool(best["passes_balanced_bar"]),
            }
        )

    return pd.DataFrame(rows)[
        [
            "selector_scope",
            "selector_key",
            "best_detector_variant",
            "TPR_sig",
            "FPR_sig",
            "score_tpr_minus_fpr",
            "passes_balanced_bar",
        ]
    ].sort_values(["selector_scope", "selector_key"]).reset_index(drop=True)


def _plot_variant_tradeoff(comparison_df: pd.DataFrame, output_path: Path) -> None:
    subset = comparison_df[comparison_df["B_bucket"] == "B>=0.2"].copy()
    markers = {
        "short_70": "o",
        "short_100": "s",
        "weak": "^",
        "mid": "D",
        "boat_drift": "P",
        "boat_transient": "X",
    }
    colors = {
        "global_tapered_fft_sig": "#1f77b4",
        "windowed_fft_sig": "#2ca02c",
        "detrended_fft_sig": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in subset.iterrows():
        ax.scatter(
            row["FPR_sig"],
            row["TPR_sig"],
            color=colors.get(str(row["detector_variant"]), "black"),
            marker=markers.get(str(row["scenario"]), "o"),
            s=60,
            alpha=0.8,
        )

    for variant, color in colors.items():
        ax.scatter([], [], color=color, marker="o", label=variant)
    for scenario, marker in markers.items():
        ax.scatter([], [], color="gray", marker=marker, label=scenario)

    ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.6, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR (B=0)")
    ax.set_ylabel("TPR (B>=0.2)")
    ax.set_title("Detector Variant Tradeoff (Significance-Calibrated)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_variant_roc_like_grid(summary_by_variant: Mapping[str, pd.DataFrame], output_path: Path) -> None:
    scenarios = sorted(
        {
            scenario
            for df in summary_by_variant.values()
            for scenario in df["scenario"].unique().tolist()
        }
    )
    if not scenarios:
        return

    ncols = 3
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.4 * nrows), squeeze=False)
    color_map = {
        "global_tapered_fft_sig": "#1f77b4",
        "windowed_fft_sig": "#2ca02c",
        "detrended_fft_sig": "#d62728",
    }

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // ncols][idx % ncols]
        for variant, summary_df in summary_by_variant.items():
            scenario_df = summary_df[summary_df["scenario"] == scenario]
            if scenario_df.empty:
                continue

            fpr = float(scenario_df.loc[scenario_df["B"] == 0.0, "false_positive_rate_sig"].mean())
            points = (
                scenario_df[scenario_df["B"] >= 0.2]
                .groupby("B", as_index=False)["recovery_rate_sig"]
                .mean()
                .sort_values("B")
            )
            for _, point in points.iterrows():
                b_value = float(point["B"])
                tpr = float(point["recovery_rate_sig"])
                ax.scatter(
                    fpr,
                    tpr,
                    color=color_map.get(variant, "black"),
                    s=48,
                    alpha=0.8,
                )
                ax.text(fpr + 0.01, tpr + 0.01, f"B={b_value:.1f}", fontsize=7, alpha=0.8)

        ax.axvline(0.1, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(0.6, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(scenario)
        ax.set_xlabel("FPR (B=0)")
        ax.set_ylabel("TPR (B>=0.2)")
        ax.grid(alpha=0.3)

    for idx in range(len(scenarios), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles = [plt.Line2D([], [], color=color_map[v], marker="o", linestyle="", label=v) for v in summary_by_variant]
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)), fontsize=8, frameon=False)
    fig.suptitle("Scenario-Level ROC-Like Grid by Detector Variant", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _evaluate_balanced_outcome(comparison_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    subset = comparison_df[
        (comparison_df["B_bucket"] == "B>=0.2")
        & (comparison_df["scenario"].isin(TARGET_SCENARIOS))
    ].copy()
    if subset.empty:
        return "FAIL", subset

    pass_counts = subset.groupby("detector_variant")["passes_balanced_bar"].sum().sort_values(ascending=False)
    best_pass = int(pass_counts.iloc[0]) if not pass_counts.empty else 0

    if best_pass >= len(TARGET_SCENARIOS):
        outcome = "PASS"
    elif best_pass > 0:
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"
    return outcome, subset.sort_values(["detector_variant", "scenario"]).reset_index(drop=True)


def _format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append("nan" if np.isnan(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_detector_variant_section(
    comparison_df: pd.DataFrame,
    outcome: str,
    balanced_subset: pd.DataFrame,
    selector_df: pd.DataFrame,
) -> str:
    compact = balanced_subset[
        ["detector_variant", "scenario", "TPR_sig", "FPR_sig", "delta_tpr_vs_baseline", "delta_fpr_vs_baseline", "passes_balanced_bar"]
    ].copy()
    table_md = _format_markdown_table(compact)
    selector_table = _format_markdown_table(selector_df) if not selector_df.empty else "_No selector rows available._"
    return f"""<!-- DETECTOR_VARIANT_SECTION_START -->
### Detector Variant Comparison
We evaluated three significance-calibrated detector variants under a shared surrogate framework: global tapered FFT, transient-window FFT, and detrended FFT. Pass/fail was pre-registered using a balanced bar for `mid` and `boat_drift`: `TPR_sig >= 0.6` at `B>=0.2` and `FPR_sig <= 0.1` at `B=0`.

Sprint 3 outcome: **{outcome}**

{table_md}

Best-Variant Selector:

{selector_table}

Artifacts:
- `outputs/detector_variants/variant_comparison.csv`
- `outputs/detector_variants/best_variant_selector.csv`
- `../figures/variant_tpr_fpr_tradeoff.png`
- `../figures/variant_roc_like_grid.png`

Interpretation is conservative: this remains methods validation and calibration, not observational proof of astrophysical QPO existence.
<!-- DETECTOR_VARIANT_SECTION_END -->
"""


def _upsert_detector_variant_section(paper_path: Path, section_md: str) -> None:
    if paper_path.exists():
        text = paper_path.read_text(encoding="utf-8")
    else:
        text = "# Gamma-Ray Burst Substructure\n\n"

    start = "<!-- DETECTOR_VARIANT_SECTION_START -->"
    end = "<!-- DETECTOR_VARIANT_SECTION_END -->"
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end)[1]
        updated = before + section_md + after
    elif "<!-- BENCHMARK_SECTION_END -->" in text:
        updated = text.replace("<!-- BENCHMARK_SECTION_END -->", section_md + "\n<!-- BENCHMARK_SECTION_END -->")
    elif "## References" in text:
        updated = text.replace("## References", section_md + "\n\n## References")
    else:
        updated = text.rstrip() + "\n\n" + section_md

    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(updated, encoding="utf-8")


def _write_decision_log(
    log_path: Path,
    config: BenchmarkConfig,
    variants: Sequence[str],
    runtime_seconds: float,
    outcome: str,
    balanced_subset: pd.DataFrame,
    selector_df: pd.DataFrame,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if balanced_subset.empty:
        result_table = "_No balanced-bar rows available._"
    else:
        result_table = _format_markdown_table(
            balanced_subset[
                [
                    "detector_variant",
                    "scenario",
                    "TPR_sig",
                    "FPR_sig",
                    "passes_balanced_bar",
                ]
            ]
        )
    selector_table = _format_markdown_table(selector_df) if not selector_df.empty else "_No selector rows available._"

    next_step = (
        "Continue targeted detector development."
        if outcome == "PASS"
        else (
            "Proceed with a mixed-result methods claim and focused detector tuning."
            if outcome == "PARTIAL"
            else "Pivot to negative/mixed methods framing and real-data bridge planning."
        )
    )

    content = f"""# Sprint 3 Decision Log

## Hypothesis
Detector variants can recover materially higher true-positive rates while preserving significance-calibrated false-positive control under shared surrogate null modeling.

## Configuration
- Timestamp: {now}
- Variants: {", ".join(variants)}
- Scenarios: {", ".join(config.scenario_names)}
- B grid: {", ".join(f"{v:.1f}" for v in config.b_grid)}
- p0 scale grid: {", ".join(f"{v:.1f}" for v in config.p0_scale_grid)}
- Replicates per cell: {config.n_replicates}
- Surrogates: {config.n_surrogates}
- Alpha: {config.alpha}
- Frequency band: {config.freq_band_min:.2f}-{config.freq_band_max:.2f} Hz
- Window padding (s): {config.window_padding_s}
- Detrend order: {config.detrend_order}
- Runtime (s): {runtime_seconds:.1f}

## Outcomes
- Sprint outcome: **{outcome}**

{result_table}

## Best-Variant Selector
{selector_table}

## Pass/Fail Rule
- Balanced bar target scenarios: `mid`, `boat_drift`
- Pass threshold: `TPR_sig >= 0.6` at `B>=0.2` and `FPR_sig <= 0.1` at `B=0`

## Next-Step Decision
{next_step}
"""
    log_path.write_text(content, encoding="utf-8")


def run_detector_variants(
    config: BenchmarkConfig,
    variants: Sequence[str],
    output_dir: Path,
    figures_dir: Path,
    paper_path: Path,
    decision_log_path: Path,
    scenario_map: Mapping[str, object] | None = None,
    max_points_for_bb: int = 2000,
    max_points_for_sig: int = 4096,
    update_paper: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    variant_list = _validate_variants(variants)
    if scenario_map is None:
        scenario_map = get_default_scenarios()

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)

    summary_by_variant: dict[str, pd.DataFrame] = {}
    run_start = time.perf_counter()

    for variant in variant_list:
        variant_cfg = replace(config, detector_variant=variant)
        variant_tmp_output = output_dir / variant
        variant_tmp_figures = figures_dir / "_variant_tmp" / variant
        run_df, summary_df = run_rigor_benchmark(
            config=variant_cfg,
            output_dir=variant_tmp_output,
            figures_dir=variant_tmp_figures,
            paper_path=paper_path,
            scenario_map=scenario_map,
            max_points_for_bb=max_points_for_bb,
            max_points_for_sig=max_points_for_sig,
            update_paper=False,
        )

        run_csv = output_dir / f"run_level_{variant}.csv"
        summary_csv = output_dir / f"summary_{variant}.csv"
        run_df.to_csv(run_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        summary_by_variant[variant] = summary_df

    comparison_df = build_variant_comparison(summary_by_variant)
    comparison_csv = output_dir / "variant_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    selector_df = build_best_variant_selector(comparison_df)
    selector_csv = output_dir / "best_variant_selector.csv"
    selector_df.to_csv(selector_csv, index=False)

    tradeoff_png = figures_dir / "variant_tpr_fpr_tradeoff.png"
    roc_like_png = figures_dir / "variant_roc_like_grid.png"
    _plot_variant_tradeoff(comparison_df, tradeoff_png)
    _plot_variant_roc_like_grid(summary_by_variant, roc_like_png)

    outcome, balanced_subset = _evaluate_balanced_outcome(comparison_df)
    runtime_seconds = time.perf_counter() - run_start
    _write_decision_log(
        log_path=decision_log_path,
        config=config,
        variants=variant_list,
        runtime_seconds=runtime_seconds,
        outcome=outcome,
        balanced_subset=balanced_subset,
        selector_df=selector_df,
    )

    if update_paper:
        section_md = _build_detector_variant_section(comparison_df, outcome, balanced_subset, selector_df)
        _upsert_detector_variant_section(paper_path, section_md)

    print("[detector-variants] complete")
    print(f"[detector-variants] variant comparison: {comparison_csv}")
    print(f"[detector-variants] selector: {selector_csv}")
    print(f"[detector-variants] figures: {tradeoff_png}, {roc_like_png}")
    print(f"[detector-variants] decision log: {decision_log_path}")
    if update_paper:
        print(f"[detector-variants] paper updated: {paper_path}")
    else:
        print("[detector-variants] paper update skipped")

    return comparison_df, summary_by_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sprint 3 detector variant benchmark with significance-calibrated scoring."
    )
    parser.add_argument("--output-dir", default="outputs/detector_variants")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--paper-path", default="paper/grb_substructure_v2.md")
    parser.add_argument("--decision-log-path", default="docs/sprint3_decision_log.md")
    parser.add_argument("--variants", default="global_tapered_fft_sig,windowed_fft_sig,detrended_fft_sig")
    parser.add_argument("--scenarios", default="short_70,short_100,weak,mid,boat_drift,boat_transient")
    parser.add_argument("--b-grid", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--p0-scales", default="0.5,1.0,2.0")
    parser.add_argument("--n-replicates", type=int, default=40)
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.02)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--n-surrogates", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--freq-band-min", type=float, default=0.35)
    parser.add_argument("--freq-band-max", type=float, default=0.45)
    parser.add_argument("--window-padding-s", type=float, default=10.0)
    parser.add_argument("--detrend-order", type=int, default=1)
    parser.add_argument("--max-points-for-bb", type=int, default=2000)
    parser.add_argument("--max-points-for-sig", type=int, default=4096)
    parser.add_argument("--skip-paper-update", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BenchmarkConfig(
        scenario_names=tuple(_parse_str_list(args.scenarios)),
        b_grid=tuple(_parse_float_list(args.b_grid)),
        p0_scale_grid=tuple(_parse_float_list(args.p0_scales)),
        n_replicates=args.n_replicates,
        freq_tolerance_hz=args.freq_tolerance_hz,
        seed_start=args.seed_start,
        n_surrogates=args.n_surrogates,
        alpha=args.alpha,
        freq_band_min=args.freq_band_min,
        freq_band_max=args.freq_band_max,
        window_padding_s=args.window_padding_s,
        detrend_order=args.detrend_order,
    )

    run_detector_variants(
        config=cfg,
        variants=tuple(_parse_str_list(args.variants)),
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        paper_path=Path(args.paper_path),
        decision_log_path=Path(args.decision_log_path),
        max_points_for_bb=args.max_points_for_bb,
        max_points_for_sig=args.max_points_for_sig,
        update_paper=not args.skip_paper_update,
    )

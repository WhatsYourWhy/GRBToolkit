from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grb_refresh import (
    BenchmarkConfig,
    adaptive_p0,
    compute_fred_rate,
    compute_knots,
    detect_qpo_signal,
    estimate_qpo_frequency,
    get_default_scenarios,
    simulate_light_curve,
    summarize_rigor_results,
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _downsample_for_bb(t: np.ndarray, counts: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if t.size <= max_points:
        return t, counts

    stride = int(np.ceil(t.size / max_points))
    usable = (counts.size // stride) * stride
    if usable <= 0:
        return t, counts

    counts_trim = counts[:usable]
    t_trim = t[:usable]
    counts_ds = counts_trim.reshape(-1, stride).sum(axis=1)
    t_ds = t_trim.reshape(-1, stride)[:, 0]
    return t_ds, counts_ds


def _format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body: list[str] = []

    for _, row in df.iterrows():
        values: list[str] = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("nan")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _build_benchmark_section(summary_df: pd.DataFrame) -> str:
    compact = summary_df[
        (summary_df["p0_scale"] == 1.0) & (summary_df["B"].isin([0.0, 0.2, 0.3, 0.4]))
    ].copy()
    compact = compact[
        [
            "scenario",
            "B",
            "recovery_rate",
            "false_positive_rate",
            "median_knots",
            "iqr_knots",
            "median_residual_gain_pct",
        ]
    ].sort_values(["scenario", "B"])

    table_md = _format_markdown_table(compact)
    return f"""<!-- BENCHMARK_SECTION_START -->
## Injection-Recovery and False-Positive Benchmark
This benchmark quantifies recovery behavior under controlled injections and explicitly separates methods validation from observational existence claims. Detection is defined as `|f0_est - f_qpo| <= 0.02 Hz`, and false-positive behavior is estimated from the `B=0` branch under matched settings.

{table_md}

### Benchmark Artifacts
- `outputs/rigor_benchmark/run_level_results.csv`
- `outputs/rigor_benchmark/recovery_summary.csv`
- `../figures/recovery_heatmap.png`
- `../figures/fpr_vs_p0.png`
- `../figures/knot_stability.png`

This section is methods validation only and should not be interpreted as direct observational proof of QPO existence in real bursts.
<!-- BENCHMARK_SECTION_END -->
"""


def _upsert_benchmark_section(paper_path: Path, section_md: str) -> None:
    if paper_path.exists():
        text = paper_path.read_text(encoding="utf-8")
    else:
        text = "# Gamma-Ray Burst Substructure\n\n"

    start = "<!-- BENCHMARK_SECTION_START -->"
    end = "<!-- BENCHMARK_SECTION_END -->"

    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end)[1]
        updated = before + section_md + after
    elif "## References" in text:
        updated = text.replace("## References", section_md + "\n\n## References")
    else:
        updated = text.rstrip() + "\n\n" + section_md

    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(updated, encoding="utf-8")


def _plot_recovery_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    heat_df = (
        summary_df.groupby(["scenario", "B"], as_index=False)["recovery_rate"].mean().pivot(
            index="scenario", columns="B", values="recovery_rate"
        )
    )
    heat_df = heat_df.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5.2))
    matrix = heat_df.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(heat_df.columns)))
    ax.set_xticklabels([f"{col:.1f}" for col in heat_df.columns], rotation=0)
    ax.set_yticks(np.arange(len(heat_df.index)))
    ax.set_yticklabels(list(heat_df.index))
    ax.set_xlabel("QPO depth B")
    ax.set_ylabel("Scenario")
    ax.set_title("Recovery Rate Heatmap (mean across p0 scales)")
    fig.colorbar(im, ax=ax, label="Recovery rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_fpr_vs_p0(summary_df: pd.DataFrame, output_path: Path) -> None:
    subset = summary_df[summary_df["B"] == 0.0].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    for scenario, group in subset.groupby("scenario"):
        group = group.sort_values("p0_scale")
        ax.plot(group["p0_scale"], group["false_positive_rate"], marker="o", label=scenario)

    ax.set_xlabel("p0 scale")
    ax.set_ylabel("False-positive rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("False-Positive Rate vs Prior Scale (B=0)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_knot_stability(summary_df: pd.DataFrame, output_path: Path) -> None:
    nonzero = sorted([val for val in summary_df["B"].unique().tolist() if val > 0.0])
    target_b = 0.3 if 0.3 in nonzero else (nonzero[-1] if nonzero else 0.0)
    subset = summary_df[summary_df["B"] == target_b].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, group in subset.groupby("scenario"):
        group = group.sort_values("p0_scale")
        ax.plot(group["p0_scale"], group["iqr_knots"], marker="o", label=scenario)

    ax.set_xlabel("p0 scale")
    ax.set_ylabel("Knot IQR")
    ax.set_title(f"Knot Stability vs Prior Scale (B={target_b:.1f})")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_rigor_benchmark(
    config: BenchmarkConfig,
    output_dir: Path,
    figures_dir: Path,
    paper_path: Path,
    scenario_map: Mapping[str, object] | None = None,
    max_points_for_bb: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_map is None:
        scenario_map = get_default_scenarios()

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)

    unknown = sorted(set(config.scenario_names) - set(scenario_map.keys()))
    if unknown:
        raise ValueError(f"Unknown scenarios in benchmark config: {', '.join(unknown)}")

    run_rows: list[dict[str, float | int | bool | str]] = []

    for s_idx, scenario_name in enumerate(config.scenario_names):
        base = scenario_map[scenario_name]
        for p_idx, p0_scale in enumerate(config.p0_scale_grid):
            for rep in range(config.n_replicates):
                seed = int(config.seed_start + s_idx * 1_000_000 + p_idx * 100_000 + rep)

                fred_only = replace(
                    base,
                    B=0.0,
                    k=0.0,
                    N=0,
                    Ai=0.0,
                    Gamma=False,
                    qpo_window_start=None,
                    qpo_window_end=None,
                    seed=seed,
                )
                t_fred, counts_fred, _, _, _ = simulate_light_curve(fred_only)
                mean_fred_rate = float(np.mean(counts_fred) / fred_only.dt)
                p0_fred = adaptive_p0(mean_fred_rate) * float(p0_scale)
                t_fred_bb, counts_fred_bb = _downsample_for_bb(t_fred, counts_fred, max_points=max_points_for_bb)
                _, fred_knots = compute_knots(t_fred_bb, counts_fred_bb, p0_fred)

                for b_value in config.b_grid:
                    params = replace(base, B=float(b_value), seed=seed)
                    t, counts, full_rate, _, _ = simulate_light_curve(params)

                    mean_rate = float(np.mean(counts) / params.dt)
                    p0 = adaptive_p0(mean_rate) * float(p0_scale)
                    t_bb, counts_bb = _downsample_for_bb(t, counts, max_points=max_points_for_bb)
                    _, knots = compute_knots(t_bb, counts_bb, p0)

                    f0_est = estimate_qpo_frequency(t, counts)
                    detected = detect_qpo_signal(f0_est, params.f_qpo, config.freq_tolerance_hz)

                    observed_rate = counts.astype(np.float64) / params.dt
                    fred_rate = compute_fred_rate(t, params)
                    residual_qpo = _rmse(observed_rate, full_rate)
                    residual_fred = _rmse(observed_rate, fred_rate)
                    edge_pct = float(0.0 if fred_knots <= 0 else ((knots - fred_knots) / fred_knots) * 100.0)

                    run_rows.append(
                        {
                            "scenario": scenario_name,
                            "B": float(b_value),
                            "p0_scale": float(p0_scale),
                            "seed": seed,
                            "knots": int(knots),
                            "edge_pct": edge_pct,
                            "f0_est": float(f0_est) if np.isfinite(f0_est) else np.nan,
                            "detected": bool(detected),
                            "residual_qpo": residual_qpo,
                            "residual_fred": residual_fred,
                        }
                    )

    run_df = pd.DataFrame(run_rows)[
        [
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
        ]
    ]
    summary_df = summarize_rigor_results(run_df)

    run_csv = output_dir / "run_level_results.csv"
    summary_csv = output_dir / "recovery_summary.csv"
    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    recovery_png = figures_dir / "recovery_heatmap.png"
    fpr_png = figures_dir / "fpr_vs_p0.png"
    knot_png = figures_dir / "knot_stability.png"
    _plot_recovery_heatmap(summary_df, recovery_png)
    _plot_fpr_vs_p0(summary_df, fpr_png)
    _plot_knot_stability(summary_df, knot_png)

    section = _build_benchmark_section(summary_df)
    _upsert_benchmark_section(paper_path, section)

    print("[rigor-benchmark] complete")
    print(f"[rigor-benchmark] run-level: {run_csv}")
    print(f"[rigor-benchmark] summary: {summary_csv}")
    print(f"[rigor-benchmark] figures: {recovery_png}, {fpr_png}, {knot_png}")
    print(f"[rigor-benchmark] paper updated: {paper_path}")

    return run_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run injection-recovery rigor benchmark for GRB scenarios.")
    parser.add_argument("--output-dir", default="outputs/rigor_benchmark")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--paper-path", default="paper/grb_substructure_v2.md")
    parser.add_argument("--scenarios", default="short_70,short_100,weak,mid,boat_drift,boat_transient")
    parser.add_argument("--b-grid", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--p0-scales", default="0.5,1.0,2.0")
    parser.add_argument("--n-replicates", type=int, default=40)
    parser.add_argument("--freq-tolerance-hz", type=float, default=0.02)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--max-points-for-bb", type=int, default=2000)
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
    )
    run_rigor_benchmark(
        config=cfg,
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        paper_path=Path(args.paper_path),
        max_points_for_bb=args.max_points_for_bb,
    )

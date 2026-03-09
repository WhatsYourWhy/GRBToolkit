from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from grb_refresh import (
    build_metrics_dataframe,
    build_paper_markdown,
    build_table1_dataframe,
    compute_aic_table,
    get_default_scenarios,
    plot_bb_sensitivity,
    plot_scenario,
    run_bb_sensitivity,
    run_scenario,
)


def _select_scenarios(all_names: List[str], requested: str | None) -> List[str]:
    if not requested:
        return all_names
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(all_names))
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return [name for name in all_names if name in selected]


def run_core_refresh(
    output_dir: Path,
    figures_dir: Path,
    paper_path: Path,
    scenario_names: Iterable[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_map = get_default_scenarios()

    results = []
    for name in scenario_names:
        result = run_scenario(name, scenario_map[name])
        results.append(result)

        fig_path = figures_dir / f"{name}.png"
        plot_scenario(result, str(fig_path))
        print(f"[core-refresh] wrote figure: {fig_path}")

    metrics_df = build_metrics_dataframe(results)
    metrics_csv = output_dir / "scenario_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    table1_df = build_table1_dataframe(results)
    table1_csv = output_dir / "table1_source.csv"
    table1_df.to_csv(table1_csv, index=False)

    representative = next((r for r in results if r.name == "boat_drift"), results[0])
    aic_df = compute_aic_table(representative)
    aic_csv = output_dir / "aic_comparison.csv"
    aic_df.to_csv(aic_csv, index=False)

    boat_ref = next((r for r in results if r.name == "boat_drift"), representative)
    sensitivity_df = run_bb_sensitivity(boat_ref)
    sensitivity_csv = output_dir / "bb_sensitivity_boat.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    sensitivity_png = figures_dir / "bb_sensitivity.png"
    plot_bb_sensitivity(sensitivity_df, str(sensitivity_png))

    metrics_rel = metrics_csv.as_posix()
    table_rel = table1_csv.as_posix()
    aic_rel = aic_csv.as_posix()
    sensitivity_rel = sensitivity_csv.as_posix()

    paper_markdown = build_paper_markdown(
        table1_df=table1_df,
        aic_df=aic_df,
        metrics_path=metrics_rel,
        table_path=table_rel,
        aic_path=aic_rel,
        sensitivity_path=sensitivity_rel,
    )
    paper_path.write_text(paper_markdown, encoding="utf-8")

    print("[core-refresh] complete")
    print(f"[core-refresh] metrics: {metrics_csv}")
    print(f"[core-refresh] table1: {table1_csv}")
    print(f"[core-refresh] aic: {aic_csv}")
    print(f"[core-refresh] bb sweep: {sensitivity_csv}")
    print(f"[core-refresh] paper: {paper_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected GRB core refresh workflow.")
    parser.add_argument(
        "--output-dir",
        default="outputs/core_refresh",
        help="Directory for CSV artifacts.",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory for scenario and sensitivity figures.",
    )
    parser.add_argument(
        "--paper-path",
        default="paper/grb_substructure_v2.md",
        help="Output markdown paper path.",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario names. Defaults to all.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario_map = get_default_scenarios()
    selected = _select_scenarios(list(scenario_map.keys()), args.scenarios)

    run_core_refresh(
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        paper_path=Path(args.paper_path),
        scenario_names=selected,
    )

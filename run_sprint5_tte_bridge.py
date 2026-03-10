from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from grb_refresh import detect_qpo_significance, phase_randomized_surrogate
from run_rigor_benchmark import _compute_significance_for_series, _downsample_for_sig, _window_slice


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return float(value)


def _manifest_required_columns() -> tuple[str, ...]:
    return ("burst_id", "input_path")


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    missing = [col for col in _manifest_required_columns() if col not in manifest_df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if manifest_df.empty:
        raise ValueError("Manifest is empty.")
    return manifest_df


def _resolve_path(raw_path: str, root_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (root_dir / p).resolve()


def _bin_event_times(event_times: np.ndarray, bin_width_s: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(event_times, dtype=np.float64)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if bin_width_s <= 0.0:
        raise ValueError("bin_width_s must be > 0")

    t_min = float(times.min())
    t_max = float(times.max())
    edges = np.arange(t_min, t_max + bin_width_s, bin_width_s, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([t_min, t_min + bin_width_s], dtype=np.float64)
    counts, bin_edges = np.histogram(times, bins=edges)
    centers = bin_edges[:-1] + (0.5 * bin_width_s)
    return centers.astype(np.float64), counts.astype(np.float64)


def _extract_time_column_from_fits(fits_path: Path) -> np.ndarray:
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = getattr(data, "names", None)
            if data is None or names is None:
                continue
            for candidate in ("TIME", "time"):
                if candidate in names:
                    return np.asarray(data[candidate], dtype=np.float64)
    raise ValueError(f"No TIME column found in FITS file: {fits_path}")


def _is_almost_uniform(t: np.ndarray, rtol: float = 0.05) -> bool:
    if t.size < 4:
        return True
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        return False
    return bool(np.all(np.abs(dt - median_dt) <= rtol * median_dt))


def _rebin_weighted_counts(t: np.ndarray, counts: np.ndarray, bin_width_s: float) -> tuple[np.ndarray, np.ndarray]:
    if t.size == 0 or counts.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    edges = np.arange(t_min, t_max + bin_width_s, bin_width_s, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([t_min, t_min + bin_width_s], dtype=np.float64)
    summed, bin_edges = np.histogram(t, bins=edges, weights=counts)
    centers = bin_edges[:-1] + (0.5 * bin_width_s)
    return centers.astype(np.float64), summed.astype(np.float64)


def _load_csv_series(csv_path: Path, bin_width_s: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    lower = {str(col).strip().lower(): col for col in df.columns}
    if "time" in lower:
        time_col = lower["time"]
    else:
        time_col = df.columns[0]

    signal_col = None
    for key in ("signal", "counts", "count", "rate"):
        if key in lower:
            signal_col = lower[key]
            break
    if signal_col is None and len(df.columns) >= 2:
        signal_col = df.columns[1]

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
    if signal_col is None:
        return _bin_event_times(t, bin_width_s=bin_width_s)

    counts = pd.to_numeric(df[signal_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(counts)
    t = t[valid]
    counts = counts[valid]
    if t.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    counts = counts[order]

    # Collapse duplicate timestamps deterministically.
    t_df = pd.DataFrame({"time": t, "counts": counts})
    grouped = t_df.groupby("time", as_index=False)["counts"].sum()
    t = grouped["time"].to_numpy(dtype=np.float64)
    counts = grouped["counts"].to_numpy(dtype=np.float64)

    if not _is_almost_uniform(t):
        t, counts = _rebin_weighted_counts(t, counts, bin_width_s=bin_width_s)
    return t.astype(np.float64), counts.astype(np.float64)


def _load_burst_series(
    burst_row: pd.Series,
    root_dir: Path,
    default_bin_width_s: float,
) -> tuple[np.ndarray, np.ndarray, str, float, Path]:
    input_path = _resolve_path(str(burst_row["input_path"]), root_dir=root_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Burst input file not found: {input_path}")

    input_type_raw = str(burst_row.get("input_type", "")).strip().lower()
    if not input_type_raw:
        suffix = input_path.suffix.lower()
        if suffix in (".fit", ".fits"):
            input_type_raw = "fits"
        elif suffix == ".csv":
            input_type_raw = "csv"
        else:
            raise ValueError(f"Cannot infer input_type for file: {input_path}")

    bin_width_s = _coerce_optional_float(burst_row.get("bin_width_s"))
    if bin_width_s is None:
        bin_width_s = float(default_bin_width_s)

    if input_type_raw == "fits":
        event_times = _extract_time_column_from_fits(input_path)
        t, counts = _bin_event_times(event_times=event_times, bin_width_s=float(bin_width_s))
    elif input_type_raw == "csv":
        t, counts = _load_csv_series(input_path, bin_width_s=float(bin_width_s))
    else:
        raise ValueError(f"Unsupported input_type '{input_type_raw}' for burst {burst_row['burst_id']}")

    return t, counts, input_type_raw, float(bin_width_s), input_path


def _null_pvalue_trials(
    counts: np.ndarray,
    dt: float,
    n_trials: int,
    n_surrogates: int,
    fmin: float,
    fmax: float,
    seed: int,
) -> np.ndarray:
    if n_trials <= 0:
        return np.array([], dtype=np.float64)
    arr = counts.astype(np.float64)
    if arr.size < 8 or not np.isfinite(dt) or dt <= 0.0:
        return np.array([], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    pvals = np.full(int(n_trials), np.nan, dtype=np.float64)
    for idx in range(int(n_trials)):
        synthetic_null = phase_randomized_surrogate(arr, rng)
        local_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        _, _, pval = _compute_significance_for_series(
            counts=synthetic_null,
            dt=float(dt),
            fmin=float(fmin),
            fmax=float(fmax),
            n_surrogates=int(n_surrogates),
            rng=np.random.default_rng(local_seed),
            statistic_mode="fft",
        )
        pvals[idx] = pval
    return pvals


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


def _plot_pvalues(run_df: pd.DataFrame, output_path: Path, alpha: float) -> None:
    if run_df.empty:
        return
    ordered = run_df.sort_values("p_value", na_position="last").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.scatter(np.arange(len(ordered)), ordered["p_value"], s=55, alpha=0.85)
    for idx, row in ordered.iterrows():
        ax.text(idx + 0.02, float(row["p_value"]) + 0.01, str(row["burst_id"]), fontsize=7)
    ax.axhline(float(alpha), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Burst index (sorted by p-value)")
    ax.set_ylabel("p-value")
    ax.set_title("Sprint 5 TTE Bridge: Burst-Level Significance p-values")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_null_calibration(run_df: pd.DataFrame, output_path: Path, alpha: float) -> None:
    if run_df.empty:
        return
    ordered = run_df.sort_values("null_empirical_fpr_alpha", na_position="last").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.bar(np.arange(len(ordered)), ordered["null_empirical_fpr_alpha"], alpha=0.85)
    ax.axhline(float(alpha), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered["burst_id"], rotation=25, ha="right")
    ax.set_ylabel("Empirical FPR under null trials")
    ax.set_title("Sprint 5 TTE Bridge: Null-Calibration Check by Burst")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_summary(run_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame(
            [
                {
                    "n_bursts": 0,
                    "n_detected_sig": 0,
                    "detected_fraction": np.nan,
                    "median_p_value": np.nan,
                    "mean_null_empirical_fpr_alpha": np.nan,
                    "alpha": float(alpha),
                    "calibration_status": "no_data",
                    "interpretation": "No bursts were processed.",
                }
            ]
        )

    n_bursts = int(len(run_df))
    n_detected = int(run_df["detected_sig"].sum())
    det_frac = float(n_detected / n_bursts)
    pvals = run_df["p_value"].to_numpy(dtype=np.float64)
    pvals = pvals[np.isfinite(pvals)]
    median_p = float(np.median(pvals)) if pvals.size > 0 else float("nan")
    null_fpr = run_df["null_empirical_fpr_alpha"].to_numpy(dtype=np.float64)
    null_fpr = null_fpr[np.isfinite(null_fpr)]
    mean_null_fpr = float(np.mean(null_fpr)) if null_fpr.size > 0 else float("nan")

    if not np.isfinite(mean_null_fpr):
        status = "unknown"
    elif mean_null_fpr <= float(alpha) * 1.5:
        status = "calibrated_or_conservative"
    else:
        status = "anti_conservative"

    return pd.DataFrame(
        [
            {
                "n_bursts": n_bursts,
                "n_detected_sig": n_detected,
                "detected_fraction": det_frac,
                "median_p_value": median_p,
                "mean_null_empirical_fpr_alpha": mean_null_fpr,
                "alpha": float(alpha),
                "calibration_status": status,
                "interpretation": "Methods calibration check only; this pilot is not observational proof of QPO existence.",
            }
        ]
    )


def _write_log(log_path: Path, run_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    compact = run_df[
        [
            "burst_id",
            "input_type",
            "n_points",
            "duration_s",
            "peak_freq_obs",
            "p_value",
            "detected_sig",
            "null_empirical_fpr_alpha",
            "detection_mode",
        ]
    ].copy()
    text = f"""# Sprint 5 Real-TTE Bridge Log

## Burst-Level Results
{_format_markdown_table(compact)}

## Pilot Summary
{_format_markdown_table(summary_df)}

## Interpretation Guardrail
- This is a methods-grade bridge and calibration report.
- These outputs are not evidence of astrophysical QPO existence by themselves.
"""
    log_path.write_text(text, encoding="utf-8")


def _build_tte_bridge_section(run_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    compact = run_df[
        [
            "burst_id",
            "p_value",
            "detected_sig",
            "peak_freq_obs",
            "null_empirical_fpr_alpha",
            "detection_mode",
        ]
    ].copy()
    return f"""<!-- TTE_BRIDGE_SECTION_START -->
## Real-TTE Bridge Pilot (Methods Validation)
We ran a small curated TTE bridge pilot using the same significance-calibrated FFT detector (phase-randomized surrogates) to evaluate portability and calibration behavior on real bursts.

{_format_markdown_table(compact)}

Summary:
{_format_markdown_table(summary_df)}

Artifacts:
- `outputs/sprint5_tte_bridge/tte_bridge_results.csv`
- `outputs/sprint5_tte_bridge/tte_bridge_summary.csv`
- `../figures/sprint5_tte_pvalues.png`
- `../figures/sprint5_tte_null_calibration.png`

Interpretation remains methods-first: this pilot does not claim observational confirmation of QPOs.
<!-- TTE_BRIDGE_SECTION_END -->
"""


def _upsert_tte_bridge_section(paper_path: Path, section_md: str) -> None:
    if paper_path.exists():
        text = paper_path.read_text(encoding="utf-8")
    else:
        text = "# Gamma-Ray Burst Substructure\n\n"

    start = "<!-- TTE_BRIDGE_SECTION_START -->"
    end = "<!-- TTE_BRIDGE_SECTION_END -->"
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


def run_sprint5_tte_bridge(
    manifest_path: Path,
    output_dir: Path,
    figures_dir: Path,
    log_path: Path,
    alpha: float = 0.05,
    n_surrogates: int = 200,
    n_null_trials: int = 80,
    freq_band_min: float = 0.35,
    freq_band_max: float = 0.45,
    default_bin_width_s: float = 0.05,
    default_window_padding_s: float = 10.0,
    max_points_for_sig: int = 4096,
    seed: int = 701000,
    update_paper: bool = False,
    paper_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root_dir = Path.cwd()
    manifest_df = _load_manifest(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if update_paper and paper_path is None:
        raise ValueError("paper_path is required when update_paper=True")

    rows: list[dict[str, object]] = []
    for idx, burst_row in manifest_df.iterrows():
        burst_id = str(burst_row["burst_id"])
        t, counts, input_type, bin_width_s, resolved_path = _load_burst_series(
            burst_row=burst_row,
            root_dir=root_dir,
            default_bin_width_s=float(default_bin_width_s),
        )

        if t.size < 8 or counts.size < 8:
            rows.append(
                {
                    "burst_id": burst_id,
                    "input_type": input_type,
                    "input_path": str(resolved_path),
                    "n_points": int(t.size),
                    "duration_s": float(np.nan),
                    "dt_median_s": float(np.nan),
                    "bin_width_s": float(bin_width_s),
                    "window_defined": False,
                    "detection_mode": "insufficient_data",
                    "peak_power_obs": np.nan,
                    "peak_freq_obs": np.nan,
                    "p_value": np.nan,
                    "p_value_global": np.nan,
                    "p_value_window": np.nan,
                    "detected_sig": False,
                    "null_trial_count": 0,
                    "null_empirical_fpr_alpha": np.nan,
                    "null_p_median": np.nan,
                    "null_p_q25": np.nan,
                    "null_p_q75": np.nan,
                    "alpha": float(alpha),
                    "freq_band_min": float(freq_band_min),
                    "freq_band_max": float(freq_band_max),
                }
            )
            continue

        t_sig_global, counts_sig_global = _downsample_for_sig(t, counts, max_points=max_points_for_sig)
        dt_global = float(np.median(np.diff(t_sig_global))) if t_sig_global.size > 1 else float(bin_width_s)
        peak_power_global, peak_freq_global, p_global = _compute_significance_for_series(
            counts=counts_sig_global.astype(np.float64),
            dt=dt_global,
            fmin=float(freq_band_min),
            fmax=float(freq_band_max),
            n_surrogates=int(n_surrogates),
            rng=np.random.default_rng(int(seed + idx * 1009 + 11)),
            statistic_mode="fft",
        )

        w_start = _coerce_optional_float(burst_row.get("qpo_window_start"))
        w_end = _coerce_optional_float(burst_row.get("qpo_window_end"))
        window_defined = bool(w_start is not None and w_end is not None and w_end > w_start)
        window_padding_s = _coerce_optional_float(burst_row.get("window_padding_s"))
        if window_padding_s is None:
            window_padding_s = float(default_window_padding_s)

        t_win, counts_win = _window_slice(
            t=t,
            counts=counts,
            start=w_start,
            end=w_end,
            padding_s=float(window_padding_s),
        )
        t_sig_window, counts_sig_window = _downsample_for_sig(t_win, counts_win, max_points=max_points_for_sig)
        dt_window = float(np.median(np.diff(t_sig_window))) if t_sig_window.size > 1 else float(bin_width_s)
        peak_power_window, peak_freq_window, p_window = _compute_significance_for_series(
            counts=counts_sig_window.astype(np.float64),
            dt=dt_window,
            fmin=float(freq_band_min),
            fmax=float(freq_band_max),
            n_surrogates=int(n_surrogates),
            rng=np.random.default_rng(int(seed + idx * 1009 + 97)),
            statistic_mode="fft",
        )

        if window_defined:
            detection_mode = "windowed"
            peak_power = peak_power_window
            peak_freq = peak_freq_window
            p_value = p_window
            counts_for_null = counts_sig_window.astype(np.float64)
            dt_for_null = dt_window
        else:
            detection_mode = "global"
            peak_power = peak_power_global
            peak_freq = peak_freq_global
            p_value = p_global
            counts_for_null = counts_sig_global.astype(np.float64)
            dt_for_null = dt_global

        detected_sig = detect_qpo_significance(p_value, alpha=alpha)
        null_pvals = _null_pvalue_trials(
            counts=counts_for_null,
            dt=float(dt_for_null),
            n_trials=int(n_null_trials),
            n_surrogates=int(n_surrogates),
            fmin=float(freq_band_min),
            fmax=float(freq_band_max),
            seed=int(seed + idx * 1009 + 211),
        )
        finite_null = null_pvals[np.isfinite(null_pvals)]
        null_emp_fpr = float(np.mean(finite_null <= float(alpha))) if finite_null.size > 0 else float("nan")

        rows.append(
            {
                "burst_id": burst_id,
                "input_type": input_type,
                "input_path": str(resolved_path),
                "n_points": int(t.size),
                "duration_s": float(t[-1] - t[0]) if t.size > 1 else float("nan"),
                "dt_median_s": float(np.median(np.diff(t))) if t.size > 1 else float("nan"),
                "bin_width_s": float(bin_width_s),
                "window_defined": bool(window_defined),
                "detection_mode": detection_mode,
                "peak_power_obs": float(peak_power) if np.isfinite(peak_power) else np.nan,
                "peak_freq_obs": float(peak_freq) if np.isfinite(peak_freq) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "p_value_global": float(p_global) if np.isfinite(p_global) else np.nan,
                "p_value_window": float(p_window) if np.isfinite(p_window) else np.nan,
                "detected_sig": bool(detected_sig),
                "null_trial_count": int(finite_null.size),
                "null_empirical_fpr_alpha": null_emp_fpr,
                "null_p_median": float(np.median(finite_null)) if finite_null.size > 0 else np.nan,
                "null_p_q25": float(np.quantile(finite_null, 0.25)) if finite_null.size > 0 else np.nan,
                "null_p_q75": float(np.quantile(finite_null, 0.75)) if finite_null.size > 0 else np.nan,
                "alpha": float(alpha),
                "freq_band_min": float(freq_band_min),
                "freq_band_max": float(freq_band_max),
            }
        )

    run_df = pd.DataFrame(rows)[
        [
            "burst_id",
            "input_type",
            "input_path",
            "n_points",
            "duration_s",
            "dt_median_s",
            "bin_width_s",
            "window_defined",
            "detection_mode",
            "peak_power_obs",
            "peak_freq_obs",
            "p_value",
            "p_value_global",
            "p_value_window",
            "detected_sig",
            "null_trial_count",
            "null_empirical_fpr_alpha",
            "null_p_median",
            "null_p_q25",
            "null_p_q75",
            "alpha",
            "freq_band_min",
            "freq_band_max",
        ]
    ].sort_values("burst_id").reset_index(drop=True)
    summary_df = _build_summary(run_df=run_df, alpha=float(alpha))

    run_csv = output_dir / "tte_bridge_results.csv"
    summary_csv = output_dir / "tte_bridge_summary.csv"
    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    pval_png = figures_dir / "sprint5_tte_pvalues.png"
    null_png = figures_dir / "sprint5_tte_null_calibration.png"
    _plot_pvalues(run_df=run_df, output_path=pval_png, alpha=float(alpha))
    _plot_null_calibration(run_df=run_df, output_path=null_png, alpha=float(alpha))
    _write_log(log_path=log_path, run_df=run_df, summary_df=summary_df)

    if update_paper:
        section_md = _build_tte_bridge_section(run_df=run_df, summary_df=summary_df)
        _upsert_tte_bridge_section(paper_path=paper_path, section_md=section_md)

    print("[sprint5-tte-bridge] complete")
    print(f"[sprint5-tte-bridge] run-level: {run_csv}")
    print(f"[sprint5-tte-bridge] summary: {summary_csv}")
    print(f"[sprint5-tte-bridge] figures: {pval_png}, {null_png}")
    print(f"[sprint5-tte-bridge] log: {log_path}")
    if update_paper:
        print(f"[sprint5-tte-bridge] paper updated: {paper_path}")
    else:
        print("[sprint5-tte-bridge] paper update skipped")
    return run_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 5 Item 5: real-TTE bridge pilot (methods/calibration framing).")
    parser.add_argument("--manifest-path", default="docs/sprint5_tte_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/sprint5_tte_bridge")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--log-path", default="docs/sprint5_tte_bridge_log.md")
    parser.add_argument("--paper-path", default="paper/grb_substructure_v2.md")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-surrogates", type=int, default=200)
    parser.add_argument("--n-null-trials", type=int, default=80)
    parser.add_argument("--freq-band-min", type=float, default=0.35)
    parser.add_argument("--freq-band-max", type=float, default=0.45)
    parser.add_argument("--default-bin-width-s", type=float, default=0.05)
    parser.add_argument("--default-window-padding-s", type=float, default=10.0)
    parser.add_argument("--max-points-for-sig", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=701000)
    parser.add_argument("--update-paper", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sprint5_tte_bridge(
        manifest_path=Path(args.manifest_path),
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        log_path=Path(args.log_path),
        alpha=float(args.alpha),
        n_surrogates=int(args.n_surrogates),
        n_null_trials=int(args.n_null_trials),
        freq_band_min=float(args.freq_band_min),
        freq_band_max=float(args.freq_band_max),
        default_bin_width_s=float(args.default_bin_width_s),
        default_window_padding_s=float(args.default_window_padding_s),
        max_points_for_sig=int(args.max_points_for_sig),
        seed=int(args.seed),
        update_paper=bool(args.update_paper),
        paper_path=Path(args.paper_path),
    )

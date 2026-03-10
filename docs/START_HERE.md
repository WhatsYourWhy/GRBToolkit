# START HERE

## What This Repo Is
GRBToolkit is a reproducible GRB timing-analysis workflow with:
- corrected flux-rate simulation (`F(t)` Poisson sampling),
- significance-calibrated detector benchmarking,
- real-TTE bridge and robustness sweeps.

## Current Claim Posture
- Locked outcome: **mixed/negative methods claim**.
- Why: calibration is controlled, but target-regime true-positive recovery did not materially improve.
- Decision references:
  - `docs/sprint4_stop_pivot_memo.md`
  - `outputs/sprint5_tte_solidify/solidify_decision.csv`
  - `docs/sprint5_publication_rationale.md`

## If You Only Run 3 Things
1. Core refresh:
   - `python run_core_refresh.py`
2. Significance benchmark:
   - `python run_rigor_benchmark.py`
3. Real-TTE bridge + robustness:
   - `python run_sprint5_tte_bridge.py --manifest-path docs/sprint5_tte_manifest.csv`
   - `python run_sprint5_tte_solidify.py --manifest-path docs/sprint5_tte_manifest.csv`

## Key Files By Role
- Simulation + scenario models:
  - `grb_refresh.py`
- Benchmark engine:
  - `run_rigor_benchmark.py`
- Detector comparison pipeline:
  - `run_detector_variants.py`
  - `run_sprint4_window_band.py`
  - `run_sprint4_welch_compare.py`
  - `run_sprint4_tiled_compare.py`
  - `run_sprint4_tiled_tune.py`
  - `run_sprint4_detrend_sweep.py`
- Real-data bridge:
  - `run_sprint5_tte_bridge.py`
  - `run_sprint5_tte_solidify.py`
- Paper draft:
  - `paper/grb_substructure_v2.md`

## Where Outputs Go
- `outputs/` for CSV artifacts by phase/sprint.
- `figures/` for manuscript and diagnostics figures.
- `docs/` for run logs, decisions, and rationale notes.

## Fast Navigation
- Docs map: `docs/DOCS_INDEX.md`
- Command catalog: `docs/COMMAND_INDEX.md`
- Sprint tracker: `docs/sprint4_candidate_experiments.md`
- Publication rationale: `docs/sprint5_publication_rationale.md`

# Command Index

## Environment
```bash
cd /mnt/c/Users/Justin/GRBToolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Core Paper Refresh
```bash
python run_core_refresh.py
```
Primary outputs:
- `outputs/core_refresh/`
- `figures/short_70.png` ... `figures/boat_transient.png`
- `paper/grb_substructure_v2.md`

## Injection-Recovery Benchmark
```bash
python run_rigor_benchmark.py
```
Primary outputs:
- `outputs/rigor_benchmark/`
- `figures/recovery_heatmap_sig.png`
- `figures/fpr_vs_p0_sig.png`

## Detector Variant Comparison (Sprint 3)
```bash
python run_detector_variants.py
```
Primary outputs:
- `outputs/detector_variants/`
- `figures/variant_tpr_fpr_tradeoff.png`
- `docs/sprint3_decision_log.md`

## Sprint 4 Detector Tracks
```bash
python run_sprint4_window_band.py
python run_sprint4_welch_compare.py
python run_sprint4_tiled_compare.py
python run_sprint4_tiled_tune.py
python run_sprint4_detrend_sweep.py
```
Primary outputs:
- `outputs/sprint4_*`
- `docs/sprint4_*_log.md`
- `docs/sprint4_stop_pivot_memo.md`

## Sprint 5 Real-TTE Bridge
```bash
python run_sprint5_tte_bridge.py --manifest-path docs/sprint5_tte_manifest.csv
```
Primary outputs:
- `outputs/sprint5_tte_bridge/`
- `figures/sprint5_tte_pvalues.png`
- `figures/sprint5_tte_null_calibration.png`

## Sprint 5 Robustness Solidify
```bash
python run_sprint5_tte_solidify.py --manifest-path docs/sprint5_tte_manifest.csv
```
Primary outputs:
- `outputs/sprint5_tte_solidify/solidify_summary_matrix.csv`
- `outputs/sprint5_tte_solidify/solidify_decision.csv`
- `figures/sprint5_tte_solidify_detected_fraction.png`
- `figures/sprint5_tte_solidify_null_fpr.png`

## Tests (WSL)
```bash
python -m pytest -q
```
Targeted tests:
```bash
python -m pytest -q tests/test_sprint5_tte_bridge.py tests/test_sprint5_tte_solidify.py
```

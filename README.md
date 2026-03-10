
# GRBToolkit – WhatsYourWhy

**Pattern is memory. Noise is a dare.**

GRBToolkit is a modular signal simulation and analysis suite designed to explore transient quasi-periodic behavior in gamma-ray burst (GRB) light curves. Built for scientific reproducibility and modular growth, it supports synthetic models (QPIX, FRED), real Fermi data, Bayesian Block sweeps, WWZ computation, and model comparison using AIC.

---

## 📂 Folder Structure

```
grbtoolkit/
├── config.py                  # Shared parameters for all models
├── qpix_model.py             # QPIX generator (oscillation + spikes + noise)
├── fred_model.py             # Canonical FRED model
├── real_data_loader.py       # CSV-based real GRB light curve loader
├── wwz_compute.py            # WWZ power map generator
├── wwz_plot_utils.py         # Visualization of WWZ maps
├── aic_compare.py            # AIC scoring utilities
├── run_qpix_sim.py           # Generates & saves QPIX signal
├── run_fred_sim.py           # Generates & saves FRED signal
├── run_realdata_sim.py       # Loads and plots real GRB CSV
├── run_bb_sweep.py           # Bayesian Block prior tuning sweep
├── run_wwz_qpix.py           # WWZ analysis on QPIX output
├── run_model_compare.py      # AIC comparison across models
├── outputs/                  # Auto-saved CSVs and plots
│   ├── qpix_signal.csv
│   ├── fred_signal.csv
│   ├── real_grb_signal.csv
│   ├── wwz_qpix_matrix.csv
│   ├── bb_sweep_*.csv
│   ├── model_comparison_aic.csv
│   └── *.png
├── data/
│   ├── grb090709A.csv
│   └── grb170817A.csv
```

---

## 🚀 Usage Instructions

### 1. Clone the repo and install requirements
Install from `requirements.txt` to keep the runtime and optional tooling in sync:

```bash
pip install -r requirements.txt
```

The file includes `numpy`, `pandas`, `matplotlib`, `astropy`, and `requests` for core usage, plus `pytest` and `autopep8` for testing and formatting.

### 2. Run a synthetic model
```bash
python run_qpix_sim.py     # QPIX model
python run_fred_sim.py     # FRED model
```

### 3. Run Bayesian Block sweep
```bash
python run_bb_sweep.py
```

### 4. Run WWZ
```bash
python run_wwz_qpix.py
```

### 5. Model comparison (AIC)
```bash
python run_model_compare.py
```

---

## 📊 Inputs + Outputs

- All `.csv` and `.png` files go to `outputs/`
- Real GRB `.csv` files (with `time` and `signal`) go in `data/`

---

## 🧠 Citations

If you use GRBToolkit, please cite:

> WhatsYourWhy *Detecting Coherent Modulation in GRB Light Curves via QPIX and Time-Frequency Simulation*. WhatsYourWhy (2025). [https://cognisi.io](https://cognisi.io)

---

## 👨‍🔬 Author


[WhatsYourWhy](https://whatsyourwhy.example.com)
[Justin@cognisi.io](mailto:Justin@cognisi.io)

---

## 🛠 Status

This toolkit is active and evolving. Feedback, replication, and respectful critique are welcome.

---

## Core Paper Refresh (Corrected Flux Model)

Use the new flux-rate refresh pipeline to regenerate paper artifacts from the corrected model (`F(t)` sampled by Poisson, not `dF/dt`).

### WSL Virtualenv Setup

```bash
cd /mnt/c/Users/Justin/GRBToolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### One-Command Refresh Run

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_core_refresh.py
```

### Generated Artifacts

- Figures: `figures/short_70.png`, `figures/short_100.png`, `figures/weak.png`, `figures/mid.png`, `figures/boat_drift.png`, `figures/boat_transient.png`, `figures/bb_sensitivity.png`
- Metrics CSV: `outputs/core_refresh/scenario_metrics.csv`
- Table source CSV: `outputs/core_refresh/table1_source.csv`
- AIC CSV: `outputs/core_refresh/aic_comparison.csv`
- BB sensitivity CSV: `outputs/core_refresh/bb_sensitivity_boat.csv`
- Paper draft: `paper/grb_substructure_v2.md`

### Rigor Benchmark Run

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_rigor_benchmark.py
```

### Rigor Benchmark Artifacts

- Run-level CSV: `outputs/rigor_benchmark/run_level_results.csv`
- Summary CSV: `outputs/rigor_benchmark/recovery_summary.csv`
- Significance recovery heatmap: `figures/recovery_heatmap_sig.png`
- Significance false-positive plot: `figures/fpr_vs_p0_sig.png`
- Knot stability plot: `figures/knot_stability.png`
- P-value distribution plot: `figures/pvalue_distribution.png`
- Paper supplement section updated in: `paper/grb_substructure_v2.md`

### Sprint 3 Detector Variant Run

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_detector_variants.py
```

### Sprint 3 Artifacts

- Variant run CSVs: `outputs/detector_variants/run_level_global_tapered_fft_sig.csv`, `outputs/detector_variants/run_level_windowed_fft_sig.csv`, `outputs/detector_variants/run_level_detrended_fft_sig.csv`
- Variant summary CSVs: `outputs/detector_variants/summary_global_tapered_fft_sig.csv`, `outputs/detector_variants/summary_windowed_fft_sig.csv`, `outputs/detector_variants/summary_detrended_fft_sig.csv`
- Variant comparison CSV: `outputs/detector_variants/variant_comparison.csv`
- Best-variant selector CSV: `outputs/detector_variants/best_variant_selector.csv`
- Tradeoff figure: `figures/variant_tpr_fpr_tradeoff.png`
- ROC-like scenario grid: `figures/variant_roc_like_grid.png`
- Decision log: `docs/sprint3_decision_log.md`
- Sprint 4 ranked follow-up plan: `docs/sprint4_candidate_experiments.md`
- Paper detector subsection updated in: `paper/grb_substructure_v2.md`

### Sprint 4 Item 1 Run (Window + Band Optimization)

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_sprint4_window_band.py
```

### Sprint 4 Item 1 Artifacts

- Candidate summary CSV: `outputs/sprint4_window_band/candidate_summary.csv`
- Best candidate CSV: `outputs/sprint4_window_band/best_candidate.csv`
- Candidate run-level CSVs: `outputs/sprint4_window_band/run_level_C*.csv`
- Candidate summary CSVs: `outputs/sprint4_window_band/summary_C*.csv`
- Score heatmap: `figures/sprint4_window_band_score.png`
- Target TPR/FPR plot: `figures/sprint4_window_band_tpr_fpr.png`
- Run log: `docs/sprint4_window_band_log.md`

### Sprint 4 Item 2 Run (Welch vs Windowed FFT)

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_sprint4_welch_compare.py
```

### Sprint 4 Item 2 Artifacts

- Comparison CSV: `outputs/sprint4_welch_compare/welch_comparison.csv`
- Decision CSV: `outputs/sprint4_welch_compare/welch_decision.csv`
- Variant run-level CSVs: `outputs/sprint4_welch_compare/run_level_windowed_fft_sig.csv`, `outputs/sprint4_welch_compare/run_level_welch_fft_sig.csv`
- Variant summary CSVs: `outputs/sprint4_welch_compare/summary_windowed_fft_sig.csv`, `outputs/sprint4_welch_compare/summary_welch_fft_sig.csv`
- Tradeoff figure: `figures/sprint4_welch_tradeoff.png`
- Run log: `docs/sprint4_welch_log.md`

### Sprint 4 Item 3 Run (Tiled Window + Multiple-Testing Control)

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_sprint4_tiled_compare.py
```

### Sprint 4 Item 3 Artifacts

- Comparison CSV: `outputs/sprint4_tiled_compare/tiled_comparison.csv`
- Decision CSV: `outputs/sprint4_tiled_compare/tiled_decision.csv`
- Variant run-level CSVs: `outputs/sprint4_tiled_compare/run_level_windowed_fft_sig.csv`, `outputs/sprint4_tiled_compare/run_level_tiled_window_fft_sig.csv`
- Variant summary CSVs: `outputs/sprint4_tiled_compare/summary_windowed_fft_sig.csv`, `outputs/sprint4_tiled_compare/summary_tiled_window_fft_sig.csv`
- Tradeoff figure: `figures/sprint4_tiled_tradeoff.png`
- Run log: `docs/sprint4_tiled_log.md`

### Sprint 4 Item 3 Tuning Sweep

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_sprint4_tiled_tune.py
```

### Sprint 4 Item 3 Tuning Artifacts

- Candidate ranking CSV: `docs/sprint4_tiled_tune/tiled_tune_candidates.csv`
- Best candidate CSV: `docs/sprint4_tiled_tune/tiled_tune_best.csv`
- Stop/Go decision CSV: `docs/sprint4_tiled_tune/tiled_tune_decision.csv`
- Baseline run-level CSV: `docs/sprint4_tiled_tune/run_level_windowed_fft_sig.csv`
- Candidate run-level CSVs: `docs/sprint4_tiled_tune/run_level_T*.csv`
- Delta scatter figure: `figures/sprint4_tiled_tune_delta_scatter.png`
- Tuning log: `docs/sprint4_tiled_tune_log.md`

### Sprint 4 Item 4 Run (Detrend Sweep with Holdout)

```bash
cd /mnt/c/Users/Justin/GRBToolkit
source .venv/bin/activate
python run_sprint4_detrend_sweep.py
```

### Sprint 4 Item 4 Artifacts

- Candidate summary CSV: `outputs/sprint4_detrend_sweep/detrend_candidate_summary.csv`
- Best candidate CSV: `outputs/sprint4_detrend_sweep/detrend_best_candidate.csv`
- Decision CSV: `outputs/sprint4_detrend_sweep/detrend_decision.csv`
- Split baseline run-level CSVs: `outputs/sprint4_detrend_sweep/run_level_train_baseline.csv`, `outputs/sprint4_detrend_sweep/run_level_holdout_baseline.csv`
- Split candidate run-level CSVs: `outputs/sprint4_detrend_sweep/run_level_train_D*.csv`, `outputs/sprint4_detrend_sweep/run_level_holdout_D*.csv`
- Tradeoff figure: `figures/sprint4_detrend_tradeoff.png`
- Delta figure: `figures/sprint4_detrend_delta.png`
- Run log: `docs/sprint4_detrend_log.md`


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
```bash
pip install numpy pandas matplotlib astropy
```

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

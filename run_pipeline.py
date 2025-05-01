
import argparse
import os
from qpix_model import generate_qpix_signal
from fred_model import generate_fred_signal
from real_data_loader import load_real_grb
from wwz_compute import compute_wwz
from wwz_plot_utils import plot_wwz_map
from logbook import log_run
import numpy as np
import pandas as pd
from datetime import datetime

def run_pipeline(model="qpix", input_file=None, run_bb=False, run_wwz=False, seed=312):
    os.makedirs("outputs", exist_ok=True)

    # Generate or load signal
    if model == "qpix":
        time, signal = generate_qpix_signal()
        output_csv = "outputs/qpix_signal.csv"
        pd.DataFrame({'time': time, 'signal': signal}).to_csv(output_csv, index=False)
        model_name = "QPIX"
        params = f"seed={seed}"
    elif model == "fred":
        time, signal = generate_fred_signal()
        output_csv = "outputs/fred_signal.csv"
        pd.DataFrame({'time': time, 'signal': signal}).to_csv(output_csv, index=False)
        model_name = "FRED"
        params = f"seed={seed}"
    elif model == "real" and input_file:
        time, signal = load_real_grb(input_file)
        output_csv = f"outputs/{Path(input_file).stem}_signal.csv"
        pd.DataFrame({'time': time, 'signal': signal}).to_csv(output_csv, index=False)
        model_name = "Real"
        params = f"input_file={input_file}"
    else:
        raise ValueError("Invalid model or missing input file.")

    segments = None

    # WWZ
    if run_wwz:
        freqs = np.linspace(0.1, 1.5, 100)
        wwz_power = compute_wwz(time, signal, freqs)
        wwz_csv = f"outputs/wwz_{model}.csv"
        wwz_png = f"outputs/wwz_{model}.png"
        pd.DataFrame(wwz_power, index=freqs, columns=time).to_csv(wwz_csv)
        plot_wwz_map(time, freqs, wwz_power, title=f"WWZ {model.upper()}", output_path=wwz_png)

    # Log it
    log_run(model=model_name, seed=seed, params=params, output_csv=output_csv, segments=segments, notes="Pipeline run")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["qpix", "fred", "real"], required=True)
    parser.add_argument("--input_file", type=str, help="Path to real GRB CSV (required for --model real)")
    parser.add_argument("--wwz", action="store_true", help="Run WWZ")
    parser.add_argument("--seed", type=int, default=312, help="Random seed (if applicable)")
    args = parser.parse_args()

    run_pipeline(model=args.model, input_file=args.input_file, run_wwz=args.wwz, seed=args.seed)

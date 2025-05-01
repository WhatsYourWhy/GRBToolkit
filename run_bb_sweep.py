
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks
from real_data_loader import load_real_grb
import os

# Optional: define adaptive p0 function
def adaptive_p0(mean_rate):
    return 0.02 * np.exp(-0.00008 * mean_rate) * 0.95

def run_bb_sweep(time, signal, prior_list, label='sample'):
    results = []
    for p0 in prior_list:
        edges = bayesian_blocks(time, signal, p0=p0)
        num_blocks = len(edges) - 1
        durations = np.diff(edges)
        avg_duration = np.mean(durations)
        results.append({
            'Label': label,
            'p0': p0,
            'NumBlocks': num_blocks,
            'AvgBlockDuration': avg_duration
        })
    return pd.DataFrame(results)

# Main runner
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Load a light curve (real or simulated)
    time, signal = load_real_grb("data/grb090709A.csv")

    # Define list of priors to test
    mean_rate = np.mean(signal)
    prior_list = [adaptive_p0(mean_rate * f) for f in [0.5, 1, 2, 4, 8]]

    # Run sweep
    df = run_bb_sweep(time, signal, prior_list, label='grb090709A')
    df.to_csv("outputs/bb_sweep_grb090709A.csv", index=False)

    # Plot sweep
    plt.plot(df['p0'], df['NumBlocks'], marker='o')
    plt.xlabel('Bayesian Block Prior (p0)')
    plt.ylabel('Number of Segments')
    plt.title('BB Sweep: grb090709A')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/bb_sweep_grb090709A.png", dpi=300)
    plt.show()

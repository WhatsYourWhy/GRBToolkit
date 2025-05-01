
import numpy as np
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
import pandas as pd

def run_bb_sweep(time, counts, prior_list, label='Sample'):
    results = []
    for p0 in prior_list:
        edges = bayesian_blocks(time, counts, p0=p0)
        num_blocks = len(edges) - 1
        block_durations = np.diff(edges)
        avg_duration = np.mean(block_durations)
        results.append({
            'Label': label,
            'Prior_p0': p0,
            'Num_Blocks': num_blocks,
            'Avg_Block_Duration': avg_duration
        })
    return pd.DataFrame(results)

# Example usage (replace with real GRB time/count data)
if __name__ == "__main__":
    duration = 10.0
    sample_rate = 1000
    time = np.linspace(0, duration, int(sample_rate * duration))
    # Simulate a light curve with noise and a few bursts
    np.random.seed(42)
    signal = np.random.poisson(5, size=len(time)).astype(float)
    signal[2000:2100] += 20  # Simulated spike
    signal[5000:5100] += 30  # Another spike

    prior_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    sweep_df = run_bb_sweep(time, signal, prior_list, label='Sim_Example')
    sweep_df.to_csv('bb_sweep_output.csv', index=False)

    # Optional: plot number of blocks vs prior
    plt.plot(sweep_df['Prior_p0'], sweep_df['Num_Blocks'], marker='o')
    plt.xlabel('BB Prior (p0)')
    plt.ylabel('Number of Segments')
    plt.title('Bayesian Block Prior Sensitivity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bb_sweep_plot.png', dpi=300)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks

# ---- Simulation Parameters ----
SIM_SEED = 312              # For reproducibility
SIM_T_START = 0.0
SIM_T_STOP = 10.0
SIM_DT = 0.005
SIM_NBINS = int((SIM_T_STOP - SIM_T_START) / SIM_DT)

FRED_A = 250.0
FRED_T0 = 0.5
FRED_TAU = 1.2
FRED_TAUR = 0.11

QPO_B = 0.38
QPO_FREQ = 0.41
QPO_PHI = 0.0

N_SPIKES = 12
SPIKE_AMP_RANGE = (20,36)
SPIKE_WIDTH = 0.025
BG_RATE = 2.2  # counts/bin

# ---- Utility Functions ----

def make_time_grid():
    return np.arange(SIM_T_START, SIM_T_STOP, SIM_DT)

def generate_spikes(t, n_spikes=N_SPIKES, amp_range=SPIKE_AMP_RANGE, width=SPIKE_WIDTH, seed=SIM_SEED):
    np.random.seed(seed)
    spike_times = np.sort(np.random.uniform(SIM_T_START+0.5, SIM_T_STOP-0.5, n_spikes))
    spike_amps = np.random.uniform(amp_range[0], amp_range[1], n_spikes)
    return spike_times, spike_amps, width

def fred_model(t):
    fred = FRED_A * np.exp(-(t-FRED_T0)/FRED_TAU) * (1.0 - np.exp(-(t-FRED_T0)/FRED_TAUR))
    fred[t<FRED_T0] = 0.0
    return fred

def qpo_model(t, base=0):
    return base + QPO_B * np.cos(2*np.pi*QPO_FREQ*t + QPO_PHI)

def spikes_model(t, spike_times, spike_amps, width):
    spikes = np.zeros_like(t)
    for ti, ai in zip(spike_times, spike_amps):
        spikes += ai * np.exp(-((t - ti) / width) ** 2)
    return spikes

def hybrid_model(t, spike_times, spike_amps, width):
    fred = fred_model(t)
    qpo = 1 + QPO_B * np.cos(2*np.pi*QPO_FREQ*t + QPO_PHI)
    spikes = spikes_model(t, spike_times, spike_amps, width)
    signal = fred * qpo + spikes + BG_RATE
    signal[signal<0] = 0
    return signal

def qpospikes_model(t, spike_times, spike_amps, width, base):
    # QPO+spikes, no envelope. Base set to match peak FRED for fair comparison.
    qpo = base + QPO_B * np.cos(2*np.pi*QPO_FREQ*t + QPO_PHI)
    spikes = spikes_model(t, spike_times, spike_amps, width)
    signal = qpo + spikes + BG_RATE
    signal[signal<0] = 0
    return signal

def simulate_counts(signal, seed=SIM_SEED):
    np.random.seed(seed + 1)
    return np.random.poisson(signal)

def run_bayesian_blocks(t, counts):
    edges = bayesian_blocks(t, counts)
    knots = len(edges) - 1
    return edges, knots

def plot_results(t, counts, model_signal, bb_edges, model_label, show=True, savefig=None):
    plt.figure(figsize=(10,4))
    plt.plot(t, counts, drawstyle='steps-mid', lw=1, color='k', alpha=0.5, label='Data (counts)')
    plt.plot(t, model_signal, lw=2, color='tab:blue', label=f'Model ({model_label})')
    for edge in bb_edges:
        plt.axvline(edge, color='r', ls='--', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Simulated Counts")
    plt.title(f"{model_label} | Bayesian Blocks: {len(bb_edges)-1} knots")
    plt.legend()
    if savefig: plt.savefig(savefig, dpi=120, bbox_inches='tight')
    if show: plt.show()

def print_stats(model_name, knot_count, counts, model_signal):
    print(f"\n--- {model_name} ---")
    print(f"Bayesian Block knots: {knot_count}")
    print(f"Total counts: {np.sum(counts):.1f}")
    print(f"Mean counts/bin: {np.mean(counts):.2f}")
    print(f"Mean model rate: {np.mean(model_signal)/SIM_DT:.2f} cts/s")
    print(f"Residual MSE: {np.mean((counts-model_signal)**2):.2f}")
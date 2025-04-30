from grb_models import *
import numpy as np

if __name__ == "__main__":
    np.random.seed(SIM_SEED)
    t = make_time_grid()
    spike_times, spike_amps, width = generate_spikes(t)
    fred = fred_model(t)
    base = np.max(fred)  # Use FRED peak for QPO+spikes baseline
    signal = qpospikes_model(t, spike_times, spike_amps, width, base)
    counts = simulate_counts(signal)
    bb_edges, knots = run_bayesian_blocks(t, counts)
    plot_results(t, counts, signal, bb_edges, "QPO+Spikes Only")
    print_stats("QPO+Spikes Only", knots, counts, signal)
    print(f"Spike params used (for later cross-sim comparison):\n  spike_times={spike_times}\n  spike_amps={spike_amps}\n  spike_width={width}")
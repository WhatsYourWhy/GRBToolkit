from grb_models import *
import numpy as np

if __name__ == "__main__":
    np.random.seed(SIM_SEED)
    t = make_time_grid()
    spike_times, spike_amps, width = generate_spikes(t)
    signal = fred_model(t) + BG_RATE
    counts = simulate_counts(signal)
    bb_edges, knots = run_bayesian_blocks(t, counts)
    plot_results(t, counts, signal, bb_edges, "FRED Only")
    print_stats("FRED Only", knots, counts, signal)
    print(f"Spike params used (for later cross-sim comparison):\n  spike_times={spike_times}\n  spike_amps={spike_amps}\n  spike_width={width}")
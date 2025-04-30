from grb_models import *
import numpy as np

if __name__ == "__main__":
    np.random.seed(SIM_SEED)
    t = make_time_grid()
    spike_times, spike_amps, width = generate_spikes(t)

    # Model signals
    fred = fred_model(t)
    base = np.max(fred)
    model_fred = fred + BG_RATE
    model_qpospikes = qpospikes_model(t, spike_times, spike_amps, width, base)
    model_hybrid = hybrid_model(t, spike_times, spike_amps, width)

    # Simulate counts
    counts_fred = simulate_counts(model_fred, seed=SIM_SEED+1)
    counts_qpospikes = simulate_counts(model_qpospikes, seed=SIM_SEED+2)
    counts_hybrid = simulate_counts(model_hybrid, seed=SIM_SEED+3)

    # Bayesian Block segmentation
    edges_f, knots_f = run_bayesian_blocks(t, counts_fred)
    edges_q, knots_q = run_bayesian_blocks(t, counts_qpospikes)
    edges_h, knots_h = run_bayesian_blocks(t, counts_hybrid)

    # Plot overlays
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    for label, x, c in zip(
        ["FRED", "QPO+Spikes", "Hybrid"],
        [counts_fred, counts_qpospikes, counts_hybrid],
        ['k','orange','purple'] ):
        plt.plot(t, x, drawstyle='steps-mid', alpha=0.5, lw=1, color=c, label=f"{label} Data")

    plt.plot(t, model_fred, lw=1.2, color="k", alpha=0.4, label="FRED Model")
    plt.plot(t, model_qpospikes, lw=1.2, color="orange", alpha=0.4, label="QPO+Spikes Model")
    plt.plot(t, model_hybrid, lw=2, color="purple", alpha=0.4, label="Hybrid Model")
    for edge in edges_h: plt.axvline(edge, color='purple', ls='--', alpha=0.15)
    for edge in edges_q: plt.axvline(edge, color='orange', ls='--', alpha=0.10)
    for edge in edges_f: plt.axvline(edge, color='k', ls='--', alpha=0.10)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title("Side-by-side: FRED vs QPO+Spikes vs Hybrid (with Bayesian Blocks)")
    plt.tight_layout()
    plt.show()

    # Print comparison table
    print("\n| Model         | Knots | Total Counts | MSE Residual | Mean Count/bin |")
    print("|--------------|-------|--------------|--------------|---------------|")
    print(f"| FRED         | {knots_f:5d} | {np.sum(counts_fred):12.1f} | {np.mean((counts_fred-model_fred)**2):12.2f} | {np.mean(counts_fred):13.2f} |")
    print(f"| QPO+Spikes   | {knots_q:5d} | {np.sum(counts_qpospikes):12.1f} | {np.mean((counts_qpospikes-model_qpospikes)**2):12.2f} | {np.mean(counts_qpospikes):13.2f} |")
    print(f"| Hybrid       | {knots_h:5d} | {np.sum(counts_hybrid):12.1f} | {np.mean((counts_hybrid-model_hybrid)**2):12.2f} | {np.mean(counts_hybrid):13.2f} |")
    print("\nSpike params used (for reproducibility across runs):")
    print(f"  spike_times={spike_times}\n  spike_amps={spike_amps}\n  spike_width={width}")
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

# Utility for extracting and plotting a lightcurve (simple binning)
def plot_lightcurve(times, binsize=0.05, tmin=None, tmax=None, grb_name='GRB'):
    if tmin is None:
        tmin = times.min()
    if tmax is None:
        tmax = times.max()
    bins = np.arange(tmin, tmax, binsize)
    counts, edges = np.histogram(times, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(10,4))
    plt.step(centers, counts, where='mid')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Counts/bin ({binsize:.3f} s)')
    plt.title(f'Raw Lightcurve: {grb_name}')
    plt.tight_layout()
    plt.show()

# --- Main analysis part ---
grb_files = {
    'GRB170817A (n5)': 'glg_tte_n5_bn170817529_v00.fit',
    'GRB090709A (n3)': 'glg_tte_n3_bn090709630_v00.fit',
    # Add more as needed
    # 'GRBXXXXXXX (nY)': 'glg_tte_nY_bnYYYYYYZZZ_v00.fit'
}

# Loop over GRBs
for grb_name, fitfile in grb_files.items():
    print(f'Processing {grb_name}: {fitfile}')
    hdul = fits.open(fitfile)
    # TTE event data are in the 3rd HDU (index 2)
    tte_data = hdul[2].data
    times = tte_data['TIME']
    # Optionally, filter on energy range using 'PHA' or 'ENERGY' columns
    # Example: times = times[(tte_data['PHA'] > lo) & (tte_data['PHA'] < hi)]
    # Simple lightcurve plot (modify bin size as you like)
    plot_lightcurve(times, binsize=0.05, grb_name=grb_name)
    # -- Insert further analysis here (QPO, spike detection, sims, etc.) --
    hdul.close()

# --- Example: add QPO or spike finder call below this point as desired

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

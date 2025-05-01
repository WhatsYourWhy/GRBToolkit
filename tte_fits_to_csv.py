
from astropy.io import fits
import numpy as np
import os

def bin_tte_fits(fits_path, csv_output_path, bin_width=0.01):
    with fits.open(fits_path) as hdul:
        data = hdul[2].data  # TTE photon data
        times = data['TIME']  # photon arrival times

        t_min = times.min()
        t_max = times.max()
        bins = np.arange(t_min, t_max + bin_width, bin_width)
        hist, edges = np.histogram(times, bins=bins)
        bin_centers = edges[:-1] + (bin_width / 2)

        output = np.column_stack((bin_centers, hist))
        np.savetxt(csv_output_path, output, delimiter=',', header='time,signal', comments='')
        print(f"Binned TTE saved to: {csv_output_path}")

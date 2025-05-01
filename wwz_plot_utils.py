
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_wwz_map(time, freqs, power, title='WWZ Power Map', output_path=None):
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(time, freqs, power, shading='auto', norm=LogNorm())
    plt.colorbar(label='WWZ Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

# You will need to integrate your WWZ computation logic separately.
# This module assumes you already have time, freqs, and power.

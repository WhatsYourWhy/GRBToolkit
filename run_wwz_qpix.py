
from qpix_model import generate_qpix_signal
from wwz_compute import compute_wwz
from wwz_plot_utils import plot_wwz_map
import numpy as np
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# Generate QPIX signal
time, signal = generate_qpix_signal()

# Frequency grid for WWZ
freqs = np.linspace(0.1, 1.5, 100)

# Compute WWZ
wwz_power = compute_wwz(time, signal, freqs)

# Save WWZ power matrix to CSV
df_wwz = pd.DataFrame(wwz_power, index=freqs, columns=time)
df_wwz.to_csv("outputs/wwz_qpix_matrix.csv")

# Save plot
plot_wwz_map(time, freqs, wwz_power, title="QPIX WWZ Power", output_path="outputs/wwz_qpix_plot.png")

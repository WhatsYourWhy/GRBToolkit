
from qpix_model import generate_qpix_signal
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Generate QPIX signal
t, s = generate_qpix_signal()

# Save signal to CSV
df = pd.DataFrame({'time': t, 'signal': s})
df.to_csv("outputs/qpix_signal.csv", index=False)

# Plot and save figure
plt.figure(figsize=(10, 4))
plt.plot(t, s, label="QPIX Signal")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.title("Synthetic QPIX Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/qpix_signal_plot.png", dpi=300)
plt.show()

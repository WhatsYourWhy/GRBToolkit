
from real_data_loader import load_real_grb
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

# Load example real data
time, signal = load_real_grb("data/grb090709A.csv")

# Save processed data
df = pd.DataFrame({'time': time, 'signal': signal})
df.to_csv("outputs/real_grb_signal.csv", index=False)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(time, signal, label="Real GRB Signal", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.title("Real GRB Light Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/real_grb_signal_plot.png", dpi=300)
plt.show()

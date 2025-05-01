
from fred_model import generate_fred_signal
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

t, s = generate_fred_signal()

df = pd.DataFrame({'time': t, 'signal': s})
df.to_csv("outputs/fred_signal.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(t, s, label="FRED Signal", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.title("Synthetic FRED Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/fred_signal_plot.png", dpi=300)
plt.show()


import numpy as np
from config import SIM_T_START, SIM_T_STOP, SIM_DT, SIM_SEED, BG_RATE

# FRED Parameters
FRED_A = 250.0
FRED_T0 = 0.5
FRED_TAU = 1.2
FRED_TAUR = 0.11

def generate_fred_signal():
    rng = np.random.default_rng(SIM_SEED)
    time = np.arange(SIM_T_START, SIM_T_STOP, SIM_DT)
    fred = FRED_A * np.exp(-(time - FRED_T0) / FRED_TAU) * (1 - np.exp(-(time - FRED_T0) / FRED_TAUR))
    fred[time < FRED_T0] = 0.0  # Enforce zero before pulse onset
    noise = rng.poisson(BG_RATE, len(time))
    signal = fred + noise
    return time, signal

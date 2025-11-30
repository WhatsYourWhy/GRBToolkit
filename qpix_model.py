
import numpy as np
from config import SIM_T_START, SIM_T_STOP, SIM_DT, SIM_SEED, QPO_FREQ, QPO_PHI, QPO_MOD_AMP, N_SPIKES, SPIKE_AMP_RANGE, SPIKE_WIDTH, BG_RATE


def generate_qpix_signal(seed: int = SIM_SEED):
    rng = np.random.default_rng(seed)
    time = np.arange(SIM_T_START, SIM_T_STOP, SIM_DT)
    modulator = QPO_MOD_AMP * np.cos(2 * np.pi * QPO_FREQ * time + QPO_PHI)

    spikes = np.zeros_like(time)
    spike_times = rng.uniform(SIM_T_START, SIM_T_STOP, N_SPIKES)
    for t0 in spike_times:
        amp = rng.uniform(*SPIKE_AMP_RANGE)
        spike = amp * np.exp(-0.5 * ((time - t0) / SPIKE_WIDTH) ** 2)
        spikes += spike

    baseline = rng.poisson(BG_RATE, len(time))
    signal = modulator + spikes + baseline
    return time, signal

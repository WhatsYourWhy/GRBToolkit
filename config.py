
# Global Simulation Configuration

SIM_SEED = 312
SIM_T_START = 0.0
SIM_T_STOP = 10.0
SIM_DT = 0.005
SIM_NBINS = int((SIM_T_STOP - SIM_T_START) / SIM_DT)

# QPIX Signal Parameters
QPO_FREQ = 0.41
QPO_PHI = 0.0
QPO_MOD_AMP = 0.38

# Spike Parameters
N_SPIKES = 12
SPIKE_AMP_RANGE = (20, 36)
SPIKE_WIDTH = 0.025

# Background Noise
BG_RATE = 2.2  # counts per bin (Poisson baseline)

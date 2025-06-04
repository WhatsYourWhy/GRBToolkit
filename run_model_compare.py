
from qpix_model import generate_qpix_signal
from fred_model import generate_fred_signal
from real_data_loader import load_real_grb
from aic_compare import compare_models
import numpy as np
import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

# Dummy log-likelihood function (replace with real one when ready)
def log_likelihood(signal, model_signal):
    residuals = signal - model_signal
    return -0.5 * np.sum((residuals)**2)

# Load QPIX signal
t_qpix, s_qpix = generate_qpix_signal()
# For now, use the signal as its own "model" (just for structure)
ll_qpix = log_likelihood(s_qpix, s_qpix)
params_qpix = 5

# Load FRED signal
t_fred, s_fred = generate_fred_signal()
ll_fred = log_likelihood(s_fred, s_fred)
params_fred = 4

# Load real GRB signal
t_real, s_real = load_real_grb("data/grb090709A.csv")
ll_real = log_likelihood(s_real, s_real)
params_real = 6

# Run AIC comparison
labels = ['QPIX', 'FRED', 'Real']
logLs = [ll_qpix, ll_fred, ll_real]
params = [params_qpix, params_fred, params_real]
df_aic = compare_models(logLs, params, labels)

df_aic.to_csv("outputs/model_comparison_aic.csv", index=False)
print(df_aic)

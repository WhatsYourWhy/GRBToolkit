
import numpy as np
import pandas as pd

def compute_aic(n_params, log_likelihood):
    return 2 * n_params - 2 * log_likelihood

def compare_models(log_likelihoods, param_counts, labels):
    """Return a dataframe of AIC scores sorted by lowest value.

    Parameters
    ----------
    log_likelihoods : list of float
        Log-likelihood of each fitted model.
    param_counts : list of int
        Number of parameters in each model.
    labels : list of str
        Human readable labels for each model.
    """

    results = []
    for label, logL, k in zip(labels, log_likelihoods, param_counts):
        aic = compute_aic(k, logL)
        results.append({'Model': label, 'LogL': logL, 'Params': k, 'AIC': aic})

    return pd.DataFrame(results).sort_values('AIC')




import numpy as np

def compute_wwz(time, signal, freqs, decay_constant=0.0125):
    N = len(time)
    T = time.max() - time.min()
    WWZ = np.zeros((len(freqs), N))

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        for j in range(N):
            t0 = time[j]
            weights = np.exp(-decay_constant * ((time - t0) ** 2))
            Sw = np.sum(weights)
            Sx = np.sum(weights * np.cos(omega * time))
            Sy = np.sum(weights * np.sin(omega * time))
            Sxx = np.sum(weights * np.cos(omega * time)**2)
            Syy = np.sum(weights * np.sin(omega * time)**2)
            Sxy = np.sum(weights * np.cos(omega * time) * np.sin(omega * time))
            Sz = np.sum(weights * signal)
            Szx = np.sum(weights * signal * np.cos(omega * time))
            Szy = np.sum(weights * signal * np.sin(omega * time))

            D = Sxx * Syy - Sxy ** 2
            if D == 0:
                WWZ[i, j] = 0
            else:
                A = (Syy * Szx - Sxy * Szy) / D
                B = (Sxx * Szy - Sxy * Szx) / D
                model = A * np.cos(omega * time) + B * np.sin(omega * time)
                residuals = signal - model
                power = np.sum(weights * residuals ** 2)
                WWZ[i, j] = 1.0 / (power + 1e-8)  # invert residual power
    return WWZ

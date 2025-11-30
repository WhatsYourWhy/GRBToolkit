import numpy as np
from fred_model import generate_fred_signal
import config


def test_generate_fred_signal_length():
    time, signal = generate_fred_signal()
    assert len(time) == len(signal) == config.SIM_NBINS


def test_generate_fred_signal_seed_controls_output():
    time_a, signal_a = generate_fred_signal(seed=789)
    time_b, signal_b = generate_fred_signal(seed=789)
    time_c, signal_c = generate_fred_signal(seed=101)

    np.testing.assert_array_equal(time_a, time_b)
    np.testing.assert_array_equal(signal_a, signal_b)
    assert np.any(signal_a != signal_c)

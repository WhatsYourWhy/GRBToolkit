import numpy as np
from qpix_model import generate_qpix_signal
import config


def test_generate_qpix_signal_length():
    time, signal = generate_qpix_signal()
    assert len(time) == len(signal) == config.SIM_NBINS


def test_generate_qpix_signal_seed_controls_output():
    time_a, signal_a = generate_qpix_signal(seed=123)
    time_b, signal_b = generate_qpix_signal(seed=123)
    time_c, signal_c = generate_qpix_signal(seed=456)

    np.testing.assert_array_equal(time_a, time_b)
    np.testing.assert_array_equal(signal_a, signal_b)
    assert np.any(signal_a != signal_c)

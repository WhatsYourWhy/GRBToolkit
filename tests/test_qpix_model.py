import numpy as np
from qpix_model import generate_qpix_signal
import config


def test_generate_qpix_signal_length():
    time, signal = generate_qpix_signal()
    assert len(time) == len(signal) == config.SIM_NBINS

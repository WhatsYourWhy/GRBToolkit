#GRBToolkit/tests/test_example.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aic_compare import compare_models

def test_always_passes():
    """
    This is a basic placeholder test.
    When you run tests, this one should always pass.
    Replace this with real tests for your project's code.
    """
    assert True == True


def test_compare_models_basic():
    logLs = [-10.0, -5.0]
    params = [3, 2]
    labels = ["A", "B"]
    df = compare_models(logLs, params, labels)
    # Model B should have the lowest AIC and appear first
    assert df.iloc[0]["Model"] == "B"


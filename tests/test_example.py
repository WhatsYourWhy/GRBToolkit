# GRBToolkit/tests/test_example.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aic_compare import compare_models


def test_compare_models_computes_expected_aic_and_order():
    log_likelihoods = [-12.5, -10.0, -8.0]
    param_counts = [2, 3, 1]
    labels = ["ModelA", "ModelB", "ModelC"]

    df = compare_models(log_likelihoods, param_counts, labels)

    expected_aics = [29.0, 26.0, 18.0]
    # Validate that AIC values are computed correctly for every model
    assert sorted(df["AIC"].tolist()) == sorted(expected_aics)

    # The models should be sorted from lowest to highest AIC
    assert df["Model"].tolist() == ["ModelC", "ModelB", "ModelA"]


def test_compare_models_handles_tied_log_likelihoods():
    log_likelihoods = [-7.0, -7.0]
    param_counts = [1, 3]
    labels = ["FewParams", "ManyParams"]

    df = compare_models(log_likelihoods, param_counts, labels)

    expected_aics = [16.0, 20.0]
    assert df["AIC"].tolist() == expected_aics
    assert df["Model"].tolist() == ["FewParams", "ManyParams"]

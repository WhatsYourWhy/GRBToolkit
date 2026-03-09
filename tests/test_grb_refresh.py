import numpy as np
import pytest

import grb_refresh
import run_core_refresh
from grb_refresh import SimulationParams, compute_flux, params_from_dict, simulate_light_curve


def _base_params_dict() -> dict:
    return {
        "A1": 100.0,
        "tau1": 2.0,
        "tau_r": 0.2,
        "B": 0.5,
        "f_qpo": 0.41,
        "phi1": 0.0,
        "N": 0,
        "Ai": 0.0,
        "R_bg": 0.0,
        "t0": 0.0,
        "k": 0.0,
        "T": 2.0,
        "dt": 0.01,
        "seed": 42,
    }


def test_compute_flux_shape_non_negative_and_pre_t0_behavior():
    raw = _base_params_dict()
    raw["t0"] = 1.0
    params = params_from_dict(raw)

    t = np.arange(0.0, params.T, params.dt)
    flux = compute_flux(t, params, np.array([]), np.array([]))

    assert flux.shape == t.shape
    assert np.all(flux >= 0.0)
    assert np.allclose(flux[t < params.t0], 0.0)


def test_simulation_is_deterministic_for_same_seed():
    params = params_from_dict(_base_params_dict())

    t_a, counts_a, _, _, _ = simulate_light_curve(params)
    t_b, counts_b, _, _, _ = simulate_light_curve(params)

    changed = grb_refresh.SimulationParams(**{**params.__dict__, "seed": params.seed + 1})
    t_c, counts_c, _, _, _ = simulate_light_curve(changed)

    np.testing.assert_array_equal(t_a, t_b)
    np.testing.assert_array_equal(counts_a, counts_b)
    assert np.any(counts_a != counts_c)


def test_validate_params_rejects_missing_or_invalid_values():
    raw = _base_params_dict()
    raw_missing = dict(raw)
    raw_missing.pop("A1")

    with pytest.raises(ValueError):
        params_from_dict(raw_missing)

    raw_bad = dict(raw)
    raw_bad["dt"] = 0.0
    with pytest.raises(ValueError):
        params_from_dict(raw_bad)


def test_no_derivative_half_wave_rectification_artifact():
    params = SimulationParams(
        A1=200.0,
        tau1=8.0,
        tau_r=0.8,
        B=0.9,
        f_qpo=0.41,
        phi1=0.0,
        N=0,
        Ai=0.0,
        R_bg=0.0,
        t0=0.0,
        k=0.0,
        T=20.0,
        dt=0.01,
        seed=7,
    )
    t = np.arange(0.0, params.T, params.dt)
    flux = compute_flux(t, params, np.array([]), np.array([]))

    # With the corrected multiplicative QPO model and B < 1, flux remains positive.
    assert np.min(flux[t > 1.0]) > 0.0


def test_core_refresh_runner_generates_expected_artifacts(tmp_path, monkeypatch):
    small_scenarios = {
        "short_70": SimulationParams(
            A1=120.0,
            tau1=0.5,
            tau_r=0.1,
            B=0.3,
            f_qpo=0.41,
            phi1=0.0,
            N=10,
            Ai=8.0,
            R_bg=15.0,
            t0=0.0,
            k=0.0,
            T=2.0,
            dt=0.01,
            seed=11,
        ),
        "boat_drift": SimulationParams(
            A1=260.0,
            tau1=8.0,
            tau_r=1.0,
            B=0.3,
            f_qpo=0.41,
            phi1=0.0,
            N=80,
            Ai=2.0,
            R_bg=20.0,
            t0=0.0,
            k=0.00001,
            T=8.0,
            dt=0.02,
            seed=22,
        ),
    }

    monkeypatch.setattr(run_core_refresh, "get_default_scenarios", lambda: small_scenarios)

    out_dir = tmp_path / "outputs"
    fig_dir = tmp_path / "figures"
    paper_path = tmp_path / "paper" / "grb_substructure_v2.md"

    run_core_refresh.run_core_refresh(
        output_dir=out_dir,
        figures_dir=fig_dir,
        paper_path=paper_path,
        scenario_names=["short_70", "boat_drift"],
    )

    assert (out_dir / "scenario_metrics.csv").exists()
    assert (out_dir / "table1_source.csv").exists()
    assert (out_dir / "aic_comparison.csv").exists()
    assert (out_dir / "bb_sensitivity_boat.csv").exists()

    assert (fig_dir / "short_70.png").exists()
    assert (fig_dir / "boat_drift.png").exists()
    assert (fig_dir / "bb_sensitivity.png").exists()

    assert paper_path.exists()
    text = paper_path.read_text(encoding="utf-8")
    assert "Recovery-Test Results" in text
    assert "../figures/short_70.png" in text

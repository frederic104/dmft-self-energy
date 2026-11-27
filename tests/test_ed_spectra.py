"""Analytic spectral limits and no-training frozen-panel evaluation."""

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from dmft_data.ed import (
    AIMParameters,
    impurity_occupancy,
    noninteracting_green_z,
    solve,
    spectral_function,
)
from dmft_data.ed_properties import dyson_green
from dmft_data.ed_spectra import evaluate_spectra, spectrum_metrics
from dmft_data.learning.causal_decoder import PoleBaseline
from dmft_data.learning.models import decode, encode
from dmft_data.provenance import sha256_file


class ConstantOutputs:
    def predict(self, x):
        # Half-filled atomic U=2: Sigma(z)=1+1/z.
        return np.tile([1., np.log(np.expm1(1.))], (len(x), 1))


class IdentityScaler:
    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def atomic_model(omega):
    model = PoleBaseline({}, 31)
    model.model = ConstantOutputs()
    model.x_scaler = model.y_scaler = IdentityScaler()
    model.poles = np.array([0.])
    basis = np.column_stack((np.ones(len(omega)), 1/(1j*omega)))
    model.basis = np.concatenate((basis.real, basis.imag))
    return model


def test_atomic_rational_spectrum_matches_ed_and_matsubara():
    params = AIMParameters(-1., 2., (-1., 1.), (0., 0.), beta=10.)
    state = solve(params)
    omega = np.arange(1, 32, 2) * np.pi / 10
    x = encode(noninteracting_green_z(params, 1j*omega)[None, :])
    model = atomic_model(omega)
    np.testing.assert_allclose(model.predict_z(x, 1j*omega), decode(model.predict(x), (len(omega),)))
    energy = np.linspace(-8, 8, 1601)
    z = energy + .05j
    predicted = -dyson_green(noninteracting_green_z(params, z)[None, :], model.predict_z(x, z)).imag / np.pi
    reference = spectral_function(state, energy, .05)[None, :]
    np.testing.assert_allclose(predicted, reference, atol=1e-12)
    metrics, _ = spectrum_metrics(predicted, reference, energy, np.array([10.]), np.array([.5]))
    assert metrics["spectrum_rmse"] < 1e-12
    assert metrics["occupation_vs_broadened_reference_rmse"] < 1e-12
    assert metrics["reference_window_weight_vs_one_rmse"] > 0
    with pytest.raises(ValueError, match="upper-half"):
        model.predict_z(x, energy)
    # Same complex-frequency G0 also covers nonzero chemical potential.
    p = AIMParameters(-.4, 0., (-1., 1.), (.2, .3), chemical_potential=.17)
    np.testing.assert_allclose(-noninteracting_green_z(p, z).imag / np.pi,
                               spectral_function(solve(p), energy, .05), atol=1e-12)


def test_frozen_spectral_pipeline(tmp_path, monkeypatch):
    import dmft_data.ed_spectra as module
    panel = tmp_path / "panel"
    (panel / "energy").mkdir(parents=True)
    omega = np.arange(1, 32, 2) * np.pi / 10
    p = AIMParameters(-1., 2., (-1., 1.), (0., 0.), beta=10.)
    path = panel / "energy/reserved.npz"
    np.savez(path, parameters=[[-1., 2., -1., 1., 0., 0., 0., 10.]],
             g0=noninteracting_green_z(p, 1j*omega)[None, :], occupancy_per_spin=[impurity_occupancy(solve(p))])
    joblib.dump({"energy_seed_31": {"causal": atomic_model(omega), "grid": SimpleNamespace()}}, panel / "panel.joblib")
    protocol = {"search_status": "closed", "evaluation_scope": "reserved_scale",
                "bundle_sha256": sha256_file(panel / "panel.joblib"),
                "reserved_datasets": {"energy": sha256_file(path)}, "source_recipe": {"axes": ["energy"]},
                "cases": [{"axis": "energy", "model_seed": 31, "key": "energy_seed_31"}], "panel": ["grid", "causal"]}
    (panel / "protocol.json").write_text(json.dumps(protocol))
    calls = []
    original = module.solve

    def counted(p):
        calls.append(p)
        return original(p)

    monkeypatch.setattr(module, "solve", counted)
    def forbidden(*args, **kwargs):
        raise AssertionError("Spectral evaluation must not train")
    monkeypatch.setattr(PoleBaseline, "fit", forbidden)
    prior = tmp_path / "prior"
    prior.mkdir()
    prior_summary = {"phase": "frozen_reserved_scale", "model_bundle_sha256": protocol["bundle_sha256"],
                     "protocol_sha256": sha256_file(panel / "protocol.json"),
                     "dataset_sha256": protocol["reserved_datasets"], "seed_summary": []}
    (prior / "summary.json").write_text(json.dumps(prior_summary))
    report = evaluate_spectra(panel, tmp_path / "spectra", points=801, etas=(.1, .2), matsubara_result=prior)
    assert report["matsubara_report_sha256"] == sha256_file(prior / "summary.json")
    assert len(calls) == 1
    assert [r["status"] for r in report["comparisons"]] == ["unsupported", "ok"] * 2
    assert report["comparisons"][1]["sigma_infinity_rmse"] < 1e-12
    assert report["comparisons"][1]["sigma_first_moment_rmse"] < 1e-12
    assert (tmp_path / "spectra/energy/example_eta_0.1.png").exists()
    assert sha256_file(panel / "panel.joblib") == protocol["bundle_sha256"]
    with pytest.raises(FileExistsError):
        evaluate_spectra(panel, tmp_path / "spectra")
    with pytest.raises(ValueError, match="Resolve"):
        evaluate_spectra(panel, tmp_path / "coarse", points=11)
    prior_summary["model_bundle_sha256"] = "wrong"
    (prior / "summary.json").write_text(json.dumps(prior_summary))
    with pytest.raises(ValueError, match="Matsubara report"):
        evaluate_spectra(panel, tmp_path / "mismatched", matsubara_result=prior)

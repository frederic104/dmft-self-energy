"""Causal head derivatives, initialization and shared-runner integration."""

import json
import warnings

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from dmft_data.ed_objective import run_comparison
from dmft_data.ed_properties import dyson_green
from dmft_data.ed_workflow import run_study
from dmft_data.learning.causal_decoder import decoder_starts
from dmft_data.learning.dyson_objective import NetworkObjective
from dmft_data.learning.models import Baseline, decode, encode
from dmft_data.provenance import sha256_file


@pytest.mark.parametrize("spacing", ["uniform", "low-frequency"])
def test_causal_head_gradient_and_positive_spectral_weights(spacing):
    omega = np.arange(1, 18, 2) * .2
    g0 = 1 / (1j * omega - np.linspace(-1, 1, 10)[:, None])
    sigma = .3 + np.linspace(.2, .5, 10)[:, None] / (1j * omega - .5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        template = Baseline({"name": "latent_mlp", "latent_dimension": 8, "max_iter": 1}, 3).fit(
            encode(g0), encode(sigma))
    before = template.predict(encode(g0)).copy()
    starts, info = decoder_starts(template, sigma, omega, seed=3, pole_count=8, pole_spacing=spacing)
    assert info["pole_spacing"] == spacing
    np.testing.assert_allclose(info["poles"], -np.array(info["poles"])[::-1])
    np.testing.assert_allclose([info["poles"][0], info["poles"][-1]], [-4, 4])
    if spacing == "low-frequency":
        assert min(abs(np.array(info["poles"]))) < 4 / 7
    pole, grid = starts["causal_latent_8"], starts["grid_latent_8"]
    np.testing.assert_array_equal(before, template.predict(encode(g0)))
    for a, b in zip(pole.model.coefs_[:-1], grid.model.coefs_[:-1]):
        np.testing.assert_array_equal(a, b)
    predicted = decode(pole.predict(encode(g0)), sigma.shape[1:])
    assert np.all(predicted.imag < 0)
    assert info["representation_fit_success"]
    objective = NetworkObjective(pole, encode(g0), sigma, g0, dyson_green(g0, sigma), 1., .001)
    vector = objective.initial
    _, grad, _ = objective.evaluate(vector)
    rng = np.random.default_rng(4)
    for _ in range(8):
        direction = rng.normal(size=len(vector))
        direction /= np.linalg.norm(direction)
        h = 1e-6
        numerical = (objective.evaluate(vector + h * direction)[0]
                     - objective.evaluate(vector - h * direction)[0]) / (2*h)
        np.testing.assert_allclose(grad @ direction, numerical, atol=1e-6, rtol=1e-4)


def test_decoder_comparison_uses_existing_runner(tmp_path):
    study = tmp_path / "study"
    run_study(study, samples=6, frequency_count=16, axes=("energy",))
    output = tmp_path / "decoder"
    summary = run_comparison(study, output, decoder_comparison=True, max_iter=3, pole_count=8)
    assert not summary["failures"]
    assert len(summary["comparisons"]) == 18
    run = output / "energy/seed_31"
    metrics = json.loads((run / "metrics.json").read_text())
    assert len(metrics["refinement"]) == 4
    assert "test_complex_rmse" not in metrics
    assert metrics["decoder_design"]["latent_dimension"] == 8
    models = joblib.load(run / "candidates.joblib")
    with np.load(study / "energy/dataset.npz") as data:
        x = encode(data["g0"][metrics["splits"]["validation"]])
    for name, model in models.items():
        if name.startswith("causal_"):
            prediction = decode(model.predict(x), (16,))
            assert np.all(prediction.imag <= 0)
            assert metrics["validation_diagnostics"][name]["positive_imaginary_fraction"] == 0
    causal = run_comparison(study, tmp_path / "causal_only", decoder_comparison=True,
                            causal_only=True, pole_spacing="low-frequency", max_iter=1)
    assert len(causal["comparisons"]) == 16
    assert not causal["failures"]


def test_representation_only_ignores_held_out_targets(tmp_path, monkeypatch):
    study = tmp_path / "study"
    run_study(study, samples=6, frequency_count=16, axes=("energy",))

    def no_training(*args, **kwargs):
        raise AssertionError("Representation-only must not train a neural network")

    monkeypatch.setattr("dmft_data.ed_objective.refine_model", no_training)
    first = run_comparison(study, tmp_path / "first", decoder_comparison=True,
                           representation_only=True, pole_spacing="low-frequency")
    assert first["recipe"]["evaluation_phase"] == "training_representation_only"
    assert not list((tmp_path / "first").rglob("*.joblib"))
    source_metrics = study / "energy/seed_31/metrics.json"
    metrics = json.loads(source_metrics.read_text())
    dataset = study / "energy/dataset.npz"
    with np.load(dataset) as data:
        arrays = {key: data[key].copy() for key in data.files}
    held_out = metrics["splits"]["validation"] + metrics["splits"]["test"]
    arrays["sigma"][held_out] += 100
    np.savez(dataset, **arrays)
    metrics["dataset_sha256"] = sha256_file(dataset)
    source_metrics.write_text(json.dumps(metrics))
    second = run_comparison(study, tmp_path / "second", decoder_comparison=True,
                            representation_only=True, pole_spacing="low-frequency")
    assert first["comparisons"] == second["comparisons"]

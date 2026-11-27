"""Gradient, matched-control and saved-study integration checks."""

import json
import warnings

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from dmft_data.ed_objective import run_comparison
from dmft_data.ed_properties import dyson_green
from dmft_data.ed_workflow import run_study
from dmft_data.learning.dyson_objective import (
    NetworkObjective,
    output_loss,
    refine_model,
)
from dmft_data.learning.models import Baseline, encode
from dmft_data.provenance import sha256_file


def training_case(name="mlp"):
    rng = np.random.default_rng(17)
    omega = np.arange(1, 8, 2)
    g0 = 1 / (1j * omega[None, :] - rng.uniform(-1, 1, (9, 1)))
    sigma = rng.uniform(.1, .3, (9, 1)) + .2 / (1j * omega[None, :] - .6)
    green = dyson_green(g0, sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Baseline({"name": name, "hidden_layers": [3], "latent_dimension": 2,
                          "max_iter": 20}, 11).fit(encode(g0), encode(sigma))
    return model, encode(g0), sigma, g0, green


@pytest.mark.parametrize("name", ["mlp", "latent_mlp"])
def test_analytic_gradient_through_dyson_and_all_network_layers(name):
    model, x, sigma, g0, green = training_case(name)
    objective = NetworkObjective(model, x, sigma, g0, green, .7, .001)
    vector = objective.initial
    loss, grad, _ = objective.evaluate(vector)
    assert np.isfinite(loss)
    rng = np.random.default_rng(8)
    for _ in range(6):
        direction = rng.normal(size=len(vector))
        direction /= np.linalg.norm(direction)
        step = 1e-6
        numeric = (objective.evaluate(vector + step * direction)[0]
                   - objective.evaluate(vector - step * direction)[0]) / (2 * step)
        np.testing.assert_allclose(grad @ direction, numeric, rtol=1e-4, atol=1e-7)
    # Check every real/imaginary output derivative independently of backprop.
    pred = objective.target + rng.normal(scale=.1, size=objective.target.shape)
    args = (objective.target, model.y_scaler, g0, green, objective.green_scale2, .7)
    _, derivative, _ = output_loss(pred, *args)
    for column in range(pred.shape[1]):
        delta = np.zeros_like(pred)
        delta[0, column] = 1e-6
        numeric = (output_loss(pred + delta, *args)[0] - output_loss(pred - delta, *args)[0]) / 2e-6
        np.testing.assert_allclose(derivative[0, column], numeric, rtol=1e-5, atol=1e-8)


def test_refinement_clones_checkpoint_and_lowers_training_objective(tmp_path):
    model, x, sigma, g0, green = training_case()
    before = model.predict(x).copy()
    for weight in (0., 1.):
        fitted, info = refine_model(model, x, sigma, g0, green, weight=weight, max_iter=15)
        assert info["final_loss"] < info["initial_loss"]
        assert info["iterations"] <= 15
        np.testing.assert_array_equal(model.predict(x), before)
        np.testing.assert_array_equal(fitted.y_scaler.scale_, model.y_scaler.scale_)
        path = tmp_path / f"model_{weight}.joblib"
        joblib.dump(fitted, path)
        np.testing.assert_array_equal(joblib.load(path).predict(x), fitted.predict(x))
    with pytest.raises(ValueError, match="Invalid refinement"):
        refine_model(model, x, sigma, g0, green, weight=-1)


def test_saved_study_refinement_ignores_held_out_property_targets(tmp_path):
    study = tmp_path / "original"
    run_study(study, samples=6, frequency_count=16, axes=("energy",))
    original_bundle = study / "energy/seed_31/candidates.joblib"
    before = sha256_file(original_bundle)
    first = tmp_path / "first"
    report = run_comparison(study, first, max_iter=2)
    assert not report["failures"]
    assert len(report["comparisons"]) == 18
    assert sha256_file(original_bundle) == before
    metrics_file = study / "energy/seed_31/metrics.json"
    metrics = json.loads(metrics_file.read_text())
    dataset = study / "energy/dataset.npz"
    with np.load(dataset) as data:
        arrays = {key: data[key].copy() for key in data.files}
    held_out = metrics["splits"]["validation"] + metrics["splits"]["test"]
    arrays["green"][held_out] += .2
    arrays["occupancy_per_spin"][held_out] += .1
    np.savez(dataset, **arrays)
    metrics["dataset_sha256"] = sha256_file(dataset)
    metrics_file.write_text(json.dumps(metrics))
    second = tmp_path / "second"
    run_comparison(study, second, max_iter=2)
    one = joblib.load(first / "energy/seed_31/candidates.joblib")
    two = joblib.load(second / "energy/seed_31/candidates.joblib")
    for name in one:
        if "_refine_" in name:
            for a, b in zip(one[name].model.coefs_, two[name].model.coefs_):
                np.testing.assert_array_equal(a, b)
    saved = json.loads((first / "energy/seed_31/metrics.json").read_text())
    assert "test_complex_rmse" not in saved
    assert saved["evaluation_phase"] == "validation_only"
    assert all(v["iterations"] <= 2 for v in saved["refinement"].values())
    with pytest.raises(FileExistsError):
        run_comparison(study, first)

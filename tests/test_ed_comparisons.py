"""Checks for the exploratory AIM split and controlled-bath workflows."""

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from dmft_data.ed import generate_small_dataset
from dmft_data.ed_baselines import EpsilonLinear, _network_diagnostics, run
from dmft_data.ed_plot import plot
from dmft_data.learning.models import Baseline, encode


def test_epsilon_linear_interpolation_extrapolation_and_duplicate_policy():
    model = EpsilonLinear().fit(np.array([2., 0., 1.]),
                                np.array([[4., -2.], [0., 0.], [1., -1.]]))
    np.testing.assert_allclose(model.predict(np.array([-1., 0., .5, 2., 3.])),
                               [[-1., 1.], [0., 0.], [.5, -.5], [4., -2.], [7., -3.]])
    assert model.coverage(np.array([-1., 0., 2., 3.])) == {
        "below": 1, "within_inclusive": 2, "above": 1}
    with pytest.raises(ValueError, match="unique"):
        EpsilonLinear().fit(np.array([0., 0.]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="at least two"):
        EpsilonLinear().fit(np.array([0.]), np.ones((1, 2)))


def test_epsilon_linear_fixed_bath_validation_and_test_isolation(tmp_path):
    data = tmp_path / "fixed.npz"
    generate_small_dataset(data, samples=30, seed=31, frequency_count=8, fixed_u=2,
                           fixed_bath=(-1, 1, .5, .7))
    result = run(data, tmp_path / "out", seed=31, split="epsilon",
                 validation_only=True, include_epsilon_linear=True)
    info = result["epsilon_linear"]
    assert info["validation_coverage"]["below"] == len(result["splits"]["validation"])
    assert "test_coverage" not in info
    assert "epsilon_linear" in result["validation_complex_rmse"]
    with np.load(data) as arrays:
        changed = {key: arrays[key].copy() for key in arrays.files}
    changed["sigma"][result["splits"]["test"]] += 50j
    np.savez(tmp_path / "changed.npz", **changed)
    other = run(tmp_path / "changed.npz", tmp_path / "other", seed=31, split="epsilon",
                validation_only=True, include_epsilon_linear=True)
    assert result["validation_complex_rmse"] == other["validation_complex_rmse"]
    assert result["selected"] == other["selected"]
    changed["parameters"][0, 2] += 1
    np.savez(tmp_path / "vary_bath.npz", **changed)
    with pytest.raises(ValueError, match="requires fixed U, bath"):
        run(tmp_path / "vary_bath.npz", tmp_path / "invalid", include_epsilon_linear=True)


def test_selected_linear_model_is_persisted_and_used_on_exploratory_test(tmp_path):
    data = tmp_path / "fixed.npz"
    generate_small_dataset(data, samples=30, seed=31, frequency_count=8, fixed_u=2,
                           fixed_bath=(-1, 1, .5, .7))
    with np.load(data) as arrays:
        linear_data = {key: arrays[key].copy() for key in arrays.files}
    linear_data["sigma"] = np.broadcast_to(
        linear_data["parameters"][:, 0, None] * (1 + 1j), linear_data["sigma"].shape
    ).copy()
    np.savez(data, **linear_data)
    result = run(data, tmp_path / "out", seed=31, include_epsilon_linear=True)
    assert result["selected"] == "epsilon_linear"
    assert result["test_complex_rmse"] < 1e-12
    assert result["epsilon_linear"]["test_coverage"]["above"] == len(result["splits"]["test"])
    model = joblib.load(tmp_path / "out" / "selected_model.joblib")
    assert type(model).__module__ == "dmft_data.learning.models"
    assert model.targets.shape[0] == len(result["splits"]["train"])


def test_controlled_bath_and_reproducible_random_split(tmp_path):
    data = tmp_path / "fixed.npz"
    generate_small_dataset(data, samples=20, seed=0, frequency_count=8, fixed_u=2,
                           fixed_bath=(-1, 1, 0.5, 0.7))
    with np.load(data) as arrays:
        np.testing.assert_array_equal(arrays["parameters"][:, 2:6],
                                      np.tile([-1, 1, 0.5, 0.7], (20, 1)))
    result = run(data, tmp_path / "random", seed=5, split="random")
    repeat = run(data, tmp_path / "repeat", seed=5, split="random", model_seed=6)
    assert result["splits"] == repeat["splits"]
    assert result["model_seed"] == 5
    assert repeat["model_seed"] == 6
    assert sum(map(len, result["splits"].values())) == 20
    assert plot(tmp_path / "random").exists()
    with pytest.raises(FileExistsError):
        plot(tmp_path / "random")


def test_varying_u_is_input_and_holdouts_are_disjoint(tmp_path):
    data = tmp_path / "vary.npz"
    generate_small_dataset(data, samples=40, seed=0, frequency_count=8)
    result = run(data, tmp_path / "vary", split="u")
    with np.load(data) as arrays:
        params = arrays["parameters"]
        assert np.all((params[result["splits"]["validation"], 1] >= 1.5) &
                      (params[result["splits"]["validation"], 1] < 2.0))
        assert np.all((params[result["splits"]["test"], 1] >= 2.0) &
                      (params[result["splits"]["test"], 1] < 2.5))
    selected = joblib.load(tmp_path / "vary" / "selected_model.joblib")
    assert selected.x_scaler.n_features_in_ == 17
    assert result["input_includes_u"] is True
    with pytest.raises(ValueError, match="U must be fixed"):
        run(data, tmp_path / "invalid", split="random")


def test_latent_candidates_use_same_training_split_and_report_diagnostics(tmp_path):
    data = tmp_path / "latent.npz"
    generate_small_dataset(data, samples=30, seed=31, frequency_count=8, fixed_u=2,
                           fixed_bath=(-1, 1, 0.5, 0.7))
    result = run(data, tmp_path / "out", seed=31, split="random", include_latent=True,
                 compare_activations=True, validation_only=True)
    expected = {"mean", "nearest", "mlp"} | {
        f"{name}_{dim}" for name in ("pca_ridge", "latent_mlp") for dim in (1, 2, 4, 8, 16)
    }
    expected |= {"mlp_tanh"} | {f"latent_mlp_tanh_{dim}" for dim in (1, 2, 4, 8, 16)}
    assert set(result["validation_complex_rmse"]) == expected
    assert result["selected"] in expected
    for dim in (1, 2, 4, 8, 16):
        assert result["validation_diagnostics"][f"latent_mlp_{dim}"][
            "actual_latent_dimension"
        ] == dim
    with np.load(data) as arrays:
        x = encode(arrays["g0"])
        reduced = {key: arrays[key].copy() for key in arrays.files}
    perturbed = {key: value.copy() for key, value in reduced.items()}
    perturbed["sigma"][result["splits"]["test"]] += 17j
    np.savez(tmp_path / "perturbed.npz", **perturbed)
    other = run(tmp_path / "perturbed.npz", tmp_path / "perturbed_out", seed=31,
                split="random", include_latent=True, compare_activations=True,
                validation_only=True)
    assert result["validation_complex_rmse"] == other["validation_complex_rmse"]
    assert result["selected"] == other["selected"]
    reduced["omega"] = reduced["omega"][:4]
    for key in ("g0", "green", "sigma"):
        reduced[key] = reduced[key][:, :4]
    np.savez(tmp_path / "short.npz", **reduced)
    with pytest.raises(ValueError, match="must support PCA dimension 16"):
        run(tmp_path / "short.npz", tmp_path / "short_out", seed=31,
            split="random", include_latent=True)
    selected = joblib.load(tmp_path / "out" / "selected_model.joblib")
    np.testing.assert_allclose(selected.x_scaler.mean_, x[result["splits"]["train"]].mean(axis=0))
    assert result["evaluation_phase"] == "validation_only"
    assert "test_real_rmse" not in result
    assert not (tmp_path / "out" / "test_predictions.npz").exists()
    import json

    diagnostics = json.loads((tmp_path / "out" / "diagnostics.json").read_text())
    assert set(diagnostics["pca_train_fit_oracle_reconstruction"]) == {"1", "2", "4", "8", "16"}
    networks = diagnostics["network_training_and_bottleneck"]
    assert len(networks["mlp"]["loss_curve"]) == result["validation_diagnostics"]["mlp"][
        "optimizer_iterations"
    ]
    assert set(networks["latent_mlp_4"]["bottleneck"]) == {"train", "validation"}
    assert networks["latent_mlp_tanh_4"]["activation"] == "tanh"
    assert "saturated_fraction_abs_ge_099" in networks["latent_mlp_tanh_4"]["bottleneck"]["train"]
    assert "unit_never_positive_count" not in networks["latent_mlp_tanh_4"]["bottleneck"]["train"]
    assert diagnostics["test_samples"] == []


def test_latent_model_has_requested_bottleneck():
    x = np.arange(80, dtype=float).reshape(20, 4) / 80
    y = np.column_stack((x[:, 0] ** 2, x[:, 1]))
    model = Baseline({"name": "latent_mlp", "latent_dimension": 4, "max_iter": 5}, 2)
    with pytest.warns():
        model.fit(x, y)
    assert model.model.hidden_layer_sizes == (64, 4, 64)
    assert model.predict(x).shape == y.shape


def test_bottleneck_diagnostics_count_observed_inactive_units():
    network = SimpleNamespace(
        loss_curve_=[1.0, 0.5], max_iter=10, tol=1e-4, n_iter_no_change=10,
        early_stopping=False, solver="adam", learning_rate_init=0.001, activation="relu",
        batch_size="auto", alpha=0.001,
        coefs_=[np.array([[1., -1.]]), np.array([[1., 0.], [0., 0.]])],
        intercepts_=[np.zeros(2), np.zeros(2)],
    )
    model = SimpleNamespace(
        model=network, x_scaler=SimpleNamespace(transform=lambda values: values),
        specification={"name": "latent_mlp"},
    )
    result = _network_diagnostics(model, np.array([[-1.], [1.]]), np.array([[-2.], [-1.]]))
    train = result["bottleneck"]["train"]
    assert train["unit_never_positive_count"] == 1
    assert train["all_units_zero_sample_fraction"] == 0.5
    assert train["positive_activation_fraction"] == 0.25
    assert train["mean_per_unit_standard_deviation"] == 0.25
    assert result["bottleneck"]["validation"]["unit_never_positive_count"] == 2
    network.activation = "tanh"
    result = _network_diagnostics(model, np.array([[-1.], [1.]]), np.array([[-2.], [-1.]]))
    train = result["bottleneck"]["train"]
    assert train["positive_activation_fraction"] == 0.25
    assert "unit_never_positive_count" not in train
    assert train["saturated_fraction_abs_ge_099"] == 0.0

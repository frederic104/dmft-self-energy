"""Checks for the exploratory AIM split and controlled-bath workflows."""

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from dmft_data.ed import generate_small_dataset
from dmft_data.ed_baselines import _network_diagnostics, run
from dmft_data.ed_plot import plot
from dmft_data.learning.models import Baseline, encode


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
    result = run(data, tmp_path / "out", seed=31, split="random", include_latent=True)
    expected = {"mean", "nearest", "mlp"} | {
        f"{name}_{dim}" for name in ("pca_ridge", "latent_mlp") for dim in (1, 2, 4, 8, 16)
    }
    assert set(result["validation_complex_rmse"]) == expected
    assert result["selected"] in expected
    for dim in (1, 2, 4, 8, 16):
        assert result["validation_diagnostics"][f"latent_mlp_{dim}"][
            "actual_latent_dimension"
        ] == dim
    with np.load(data) as arrays:
        x = encode(arrays["g0"])
        reduced = {key: arrays[key].copy() for key in arrays.files}
    reduced["omega"] = reduced["omega"][:4]
    for key in ("g0", "green", "sigma"):
        reduced[key] = reduced[key][:, :4]
    np.savez(tmp_path / "short.npz", **reduced)
    with pytest.raises(ValueError, match="must support PCA dimension 16"):
        run(tmp_path / "short.npz", tmp_path / "short_out", seed=31,
            split="random", include_latent=True)
    selected = joblib.load(tmp_path / "out" / "selected_model.joblib")
    np.testing.assert_allclose(selected.x_scaler.mean_, x[result["splits"]["train"]].mean(axis=0))
    assert np.isfinite(result["test_real_rmse"])
    assert 0 <= result["test_positive_imaginary_fraction"] <= 1
    import json

    diagnostics = json.loads((tmp_path / "out" / "diagnostics.json").read_text())
    assert set(diagnostics["pca_train_fit_oracle_reconstruction"]) == {"1", "2", "4", "8", "16"}
    networks = diagnostics["network_training_and_bottleneck"]
    assert len(networks["mlp"]["loss_curve"]) == result["validation_diagnostics"]["mlp"][
        "optimizer_iterations"
    ]
    assert set(networks["latent_mlp_4"]["bottleneck"]) == {"train", "validation"}
    assert [row["index"] for row in diagnostics["test_samples"]] == result["splits"]["test"]
    with np.load(tmp_path / "out" / "test_predictions.npz") as saved:
        err = (saved["prediction"] - saved["reference"]).real
        np.testing.assert_allclose(
            [row["oracle_real_offset"] for row in diagnostics["test_samples"]],
            err.mean(axis=1),
        )


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
        early_stopping=False, solver="adam", learning_rate_init=0.001,
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

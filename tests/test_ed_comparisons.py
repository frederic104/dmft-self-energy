"""Checks for the exploratory AIM split and controlled-bath workflows."""

import joblib
import numpy as np
import pytest

from dmft_data.ed import generate_small_dataset
from dmft_data.ed_baselines import run
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
    repeat = run(data, tmp_path / "repeat", seed=5, split="random")
    assert result["splits"] == repeat["splits"]
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
        f"{name}_{dim}" for name in ("pca_ridge", "latent_mlp") for dim in (4, 8, 16)
    }
    assert set(result["validation_complex_rmse"]) == expected
    assert result["selected"] in expected
    for dim in (4, 8, 16):
        assert result["validation_diagnostics"][f"latent_mlp_{dim}"][
            "actual_latent_dimension"
        ] == dim
    with np.load(data) as arrays:
        x = encode(arrays["g0"])
    selected = joblib.load(tmp_path / "out" / "selected_model.joblib")
    np.testing.assert_allclose(selected.x_scaler.mean_, x[result["splits"]["train"]].mean(axis=0))
    assert np.isfinite(result["test_real_rmse"])
    assert 0 <= result["test_positive_imaginary_fraction"] <= 1


def test_latent_model_has_requested_bottleneck():
    x = np.arange(80, dtype=float).reshape(20, 4) / 80
    y = np.column_stack((x[:, 0] ** 2, x[:, 1]))
    model = Baseline({"name": "latent_mlp", "latent_dimension": 4, "max_iter": 5}, 2)
    with pytest.warns():
        model.fit(x, y)
    assert model.model.hidden_layer_sizes == (64, 4, 64)
    assert model.predict(x).shape == y.shape

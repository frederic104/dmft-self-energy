"""Checks for the exploratory AIM split and controlled-bath workflows."""

import joblib
import numpy as np
import pytest

from dmft_data.ed import generate_small_dataset
from dmft_data.ed_baselines import run
from dmft_data.ed_plot import plot


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

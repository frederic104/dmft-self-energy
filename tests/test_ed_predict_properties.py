"""Reference-free AIM property reconstruction checks."""

import json

import numpy as np
import pytest
from scipy.special import expit

from dmft_data.ed import matsubara_frequencies
from dmft_data.ed_predict_properties import reconstruct, run
from dmft_data.provenance import sha256_file


def test_reconstruct_shifted_noninteracting_levels(tmp_path):
    beta = 20.
    omega = matsubara_frequencies(beta, 64)
    levels = np.array([-.2, .1])
    g0 = 1 / (1j * omega - levels[:, None])
    sigma = np.full_like(g0, .03)
    source = tmp_path / "input.npz"
    np.savez(source, g0=g0, sigma=sigma, omega=omega, beta=beta)
    before = sha256_file(source)
    output = tmp_path / "output"
    summary = run(source, output)
    with np.load(output / "properties.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["green"], 1 / (1j * omega - levels[:, None] - .03))
        np.testing.assert_allclose(saved["occupancy_tail"], expit(-beta * (levels + .03)), atol=1e-7)
    assert summary["input_sha256"] == before == sha256_file(source)
    assert summary == json.loads((output / "summary.json").read_text())
    assert str(tmp_path) not in json.dumps(summary)
    assert summary["occupancy_tail_out_of_bounds_count"] == 0
    with pytest.raises(FileExistsError):
        run(source, output)


def test_invalid_mesh_or_shapes_rejected():
    omega = matsubara_frequencies(20., 32)
    g0 = (1 / (1j * omega - .1))[None, :]
    with pytest.raises(ValueError, match="state-by-frequency"):
        reconstruct(g0.real, g0, omega, 20.)
    with pytest.raises(ValueError, match="state-by-frequency"):
        reconstruct(g0, g0[:, :16], omega, 20.)
    with pytest.raises(ValueError, match="consecutive"):
        reconstruct(g0, np.zeros_like(g0), omega[::-1], 20.)
    with pytest.raises(ValueError, match="Singular"):
        reconstruct(g0, 1 / g0, omega, 20.)

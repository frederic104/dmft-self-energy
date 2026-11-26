"""Independent limiting-case and consistency checks for the finite-bath ED reference."""

import json

import numpy as np
import pytest

from dmft_data.ed import (
    AIMParameters,
    generate_small_dataset,
    green_iw,
    impurity_occupancy,
    matsubara_frequencies,
    noninteracting_green_iw,
    self_energy_iw,
    solve,
    spectral_function,
)


def test_noninteracting_ed_matches_analytic_g0_and_zero_self_energy():
    params = AIMParameters(
        -0.3, 0.0, (-1.1, 0.8), (0.7, 0.4), chemical_potential=0.23, beta=11.0
    )
    omega = matsubara_frequencies(params.beta, 24)
    result = solve(params)
    np.testing.assert_allclose(
        green_iw(result, omega), noninteracting_green_iw(params, omega), atol=1e-12
    )
    np.testing.assert_allclose(self_energy_iw(result, omega), 0.0, atol=1e-11)


def test_atomic_half_filled_self_energy_is_exact():
    interaction = 2.4
    params = AIMParameters(
        -interaction / 2, interaction, (-1.0, 1.0), (0.0, 0.0), beta=9.0
    )
    omega = matsubara_frequencies(params.beta, 20)
    sigma = self_energy_iw(solve(params), omega)
    expected = interaction / 2 + interaction**2 / (4j * omega)
    np.testing.assert_allclose(sigma, expected, atol=1e-11)


@pytest.mark.parametrize("epsilon, interaction, mu, beta", [
    (-0.4, 2.1, 0.3, 3.0),
    (0.7, 1.5, -0.2, 1.7),
    (-1.2, 3.0, -0.5, 4.0),
])
def test_asymmetric_atomic_green_and_occupancy(epsilon, interaction, mu, beta):
    params = AIMParameters(epsilon, interaction, (-1.0, 1.0), (0.0, 0.0), mu, beta)
    result = solve(params)
    single = np.exp(-beta * (epsilon - mu))
    double = np.exp(-beta * (2 * (epsilon - mu) + interaction))
    occupation = (single + double) / (1 + 2 * single + double)
    omega = matsubara_frequencies(beta, 20)
    z = 1j * omega + mu - epsilon
    expected = (1 - occupation) / z + occupation / (z - interaction)
    for spin in (0, 1):
        np.testing.assert_allclose(impurity_occupancy(result, spin), occupation, atol=1e-12)
        np.testing.assert_allclose(green_iw(result, omega, spin), expected, atol=1e-12)
    np.testing.assert_allclose(impurity_occupancy(result), occupation, atol=1e-12)


def test_interacting_high_frequency_self_energy_moments():
    params = AIMParameters(-0.7, 2.6, (-1.2, 0.9), (0.5, 0.8), 0.25, 8.0)
    result = solve(params)
    occupation = impurity_occupancy(result)
    omega = np.array([2000.0, 4000.0])
    sigma = self_energy_iw(result, omega)
    hartree = params.interaction_u * occupation
    first_moment = params.interaction_u**2 * occupation * (1 - occupation)
    np.testing.assert_allclose(sigma.real, hartree, atol=2e-6, rtol=0)
    moment_errors = np.abs(-omega * sigma.imag - first_moment)
    assert moment_errors[1] < moment_errors[0]
    np.testing.assert_allclose(-omega * sigma.imag, first_moment, atol=5e-6, rtol=0)


def test_spectral_sum_rule_and_matsubara_causality():
    params = AIMParameters(-0.6, 3.0, (-1.3, 1.1), (0.65, 0.45), beta=16.0)
    result = solve(params)
    energy = np.linspace(-12.0, 12.0, 10001)
    spectrum = spectral_function(result, energy, broadening=0.04)
    assert np.all(spectrum >= 0)
    integral = np.sum((spectrum[1:] + spectrum[:-1]) * np.diff(energy) / 2)
    assert abs(integral - 1.0) < 0.015
    sigma = self_energy_iw(result, matsubara_frequencies(params.beta, 32))
    assert np.all(sigma.imag <= 1e-11)


def test_generator_is_reproducible_and_keeps_metadata(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    generate_small_dataset(first, samples=3, seed=12, frequency_count=7)
    generate_small_dataset(second, samples=3, seed=12, frequency_count=7)
    with np.load(first) as a, np.load(second) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        assert json.loads(str(a["metadata"]))["seed"] == 12
    with pytest.raises(FileExistsError):
        generate_small_dataset(first)
    with pytest.raises(ValueError, match=".npz"):
        generate_small_dataset(tmp_path / "missing_extension")


def test_fixed_u_generator_and_blocked_baseline(tmp_path):
    from dmft_data.ed_baselines import run

    dataset = tmp_path / "fixed.npz"
    generate_small_dataset(dataset, samples=18, seed=0, frequency_count=8, fixed_u=2.0)
    with np.load(dataset) as data:
        assert np.all(data["parameters"][:, 1] == 2.0)
        levels = data["parameters"][:, 0].copy()
    metrics = run(dataset, tmp_path / "results")
    splits = metrics["splits"]
    assert np.all(levels[splits["validation"]] < -0.6)
    assert np.all(levels[splits["train"]] >= -0.6)
    assert np.all(levels[splits["train"]] < -0.2)
    assert np.all(levels[splits["test"]] >= -0.2)
    assert (tmp_path / "results" / "selected_model.joblib").exists()
    with pytest.raises(FileExistsError):
        run(dataset, tmp_path / "results")

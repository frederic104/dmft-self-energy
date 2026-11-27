"""Analytic property checks and an integrated exploratory AIM run."""

import json

import numpy as np
import pytest

from dmft_data.ed import (
    AIMParameters,
    green_iw,
    impurity_occupancy,
    matsubara_frequencies,
    solve,
)
from dmft_data.ed_properties import compare_properties, dyson_green, occupancy_estimates
from dmft_data.ed_validation import analyze
from dmft_data.ed_workflow import run_study
from dmft_data.provenance import sha256_file


def test_density_tail_matches_noninteracting_fermi_function():
    beta, energy = 8., .7
    omega = matsubara_frequencies(beta, 64)
    green = (1 / (1j * omega - energy))[None, :]
    exact = 1 / (1 + np.exp(beta * energy))
    raw, corrected = occupancy_estimates(green, omega, beta)
    assert abs(corrected[0] - exact) < 1e-7
    assert abs(corrected[0] - exact) < abs(raw[0] - exact)
    np.testing.assert_allclose(dyson_green(green, np.zeros_like(green)), green)
    with pytest.raises(ValueError, match="Singular"):
        dyson_green(green, 1 / green)
    with pytest.raises(ValueError, match="at least 16"):
        occupancy_estimates(green[:, :8], omega[:8], beta)


def test_interacting_atomic_density_and_dyson_error_propagation():
    beta, epsilon, interaction = 5., -.4, 2.1
    params = AIMParameters(epsilon, interaction, (-1., 1.), (0., 0.), beta=beta)
    solved = solve(params)
    single, double = np.exp(-beta * epsilon), np.exp(-beta * (2 * epsilon + interaction))
    exact = np.array([(single + double) / (1 + 2 * single + double)])
    np.testing.assert_allclose(exact[0], impurity_occupancy(solved), atol=1e-12)
    omega = matsubara_frequencies(beta, 64)
    green = green_iw(solved, omega)[None, :]
    g0 = (1 / (1j * omega - epsilon))[None, :]
    sigma = 1 / g0 - 1 / green
    metrics, arrays = compare_properties(g0, sigma, green, exact, omega, beta)
    assert metrics["green_complex_rmse"] < 1e-14
    assert metrics["reference_tail_vs_exact_rmse"] < 1e-5
    assert metrics["occupancy_tail_model_vs_reference_method_rmse"] < 1e-12
    perturbed, _ = compare_properties(g0, sigma + .2, green, exact, omega, beta)
    assert perturbed["green_complex_rmse"] > .001
    assert perturbed["occupancy_tail_vs_exact_rmse"] > metrics["occupancy_tail_vs_exact_rmse"]
    assert arrays["green"].shape == green.shape


def test_workflow_reuses_models_for_coupling_energy_and_properties(tmp_path):
    output = tmp_path / "study"
    summary = run_study(output, samples=6, frequency_count=32)
    assert len(summary["comparisons"]) == 28
    assert all(row["property_status"] == "ok" for row in summary["comparisons"])
    assert (output / "comparison.csv").exists()
    assert (output / "recipe.json").exists()
    with pytest.raises(FileExistsError):
        run_study(output)
    for axis in ("coupling", "energy"):
        dataset = output / axis / "dataset.npz"
        result = output / axis / "seed_31"
        metrics = json.loads((result / "metrics.json").read_text())
        assert metrics["bath_holdout"]["axis"] == axis
        assert "test_complex_rmse" not in metrics
        assert not (result / "test_predictions.npz").exists()
        with np.load(dataset) as data:
            arrays = {key: data[key].copy() for key in data.files}
        p, scales = arrays["parameters"], arrays["bath_scale"]
        varying, fixed = (slice(4, 6), slice(2, 4)) if axis == "coupling" else (slice(2, 4), slice(4, 6))
        expected_base = np.array([.5, .7] if axis == "coupling" else [-1., 1.])
        np.testing.assert_allclose(p[:, varying], scales[:, None] * expected_base)
        np.testing.assert_array_equal(p[:, fixed], np.tile(p[0, fixed], (len(p), 1)))
        if axis == "energy":
            w = arrays["omega"]
            expected_g0 = 1 / (1j*w - p[:, 0, None] - .5**2/(1j*w + scales[:, None])
                               - .7**2/(1j*w - scales[:, None]))
            np.testing.assert_allclose(arrays["g0"], expected_g0, atol=1e-14)
        with np.load(result / "analysis" / "validation_properties.npz") as saved:
            assert saved["indices"].tolist() == metrics["splits"]["validation"]
            assert saved["ridge_green"].shape == (6, 32)
        # Held-out property targets cannot alter validation diagnostics.
        arrays["green"][metrics["splits"]["test"]] += 100
        arrays["occupancy_per_spin"][metrics["splits"]["test"]] = 100
        changed = output / axis / "changed.npz"
        np.savez(changed, **arrays)
        metrics["dataset_sha256"] = sha256_file(changed)
        (result / "metrics.json").write_text(json.dumps(metrics))
        repeat = analyze(changed, result, result / "repeat", properties=True)
        original = json.loads((result / "analysis" / "summary.json").read_text())
        assert repeat["properties"] == original["properties"]

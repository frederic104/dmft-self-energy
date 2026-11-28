"""Analytic saved-result fixtures: no training or frozen model evaluation."""

import csv
import json

import numpy as np
import pytest
from scipy.special import expit

from dmft_data.ed import matsubara_frequencies
from dmft_data.ed_properties import compare_properties, dyson_errors, dyson_green, occupancy_estimates
from dmft_data.ed_report import aggregate, counterfactual_properties, report
from dmft_data.ed_spectra import spectrum_metrics
from dmft_data.provenance import sha256_file


def test_exact_dyson_identity_and_linear_remainder():
    g0 = np.array([[1/(.2j-.3), 1/(2j-.3)]])
    sigma = np.zeros_like(g0)
    for size in (0., 1e-5, .8):
        predicted = sigma + size
        green = dyson_green(g0, predicted)
        e = dyson_errors(sigma, predicted, g0, green)
        np.testing.assert_allclose(e["identity_residual"], 0, atol=1e-15)
        np.testing.assert_allclose(e["linear_remainder"],
                                   g0 * size * e["green_error"], atol=1e-15)
        if size:
            np.testing.assert_allclose(abs(e["linear_remainder"]/e["green_error"]),
                                       e["expansion_parameter"], rtol=1e-6)
    bad = dyson_errors(sigma, predicted, g0+.1, green)
    assert np.max(abs(bad["identity_residual"])) > .01
    with pytest.raises(ValueError, match="common mesh"):
        dyson_errors(sigma, predicted[:, :1], g0, green)


def test_seed_sd_is_population_and_failures_are_not_zero():
    rows = [{"candidate": "a", "model_seed": s, "status": "ok", "sigma_rmse": v}
            for s, v in ((31, 1.), (32, 3.))]
    rows += [{"candidate": "a", "model_seed": 33, "status": "failed"},
             {"candidate": "b", "model_seed": 31, "status": "unsupported"}]
    result = aggregate(rows)
    assert len(result) == 1
    assert result[0]["mean"] == 2
    assert result[0]["std"] == 1
    assert result[0]["count"] == 2 and result[0]["expected_rows"] == 3
    assert aggregate(rows[:1])[0]["std"] is None


def test_counterfactual_decomposition_and_nonlinear_interaction():
    beta = 20.
    omega = matsubara_frequencies(beta, 64)
    g0 = (1/(1j*omega-.2))[None, :]
    sigma = np.zeros_like(g0)
    reference_n = np.array([expit(-beta*.2)])
    prediction = sigma + .05 + .03/(1j*omega-.7)
    cases, offset, interaction_g, interaction_n = counterfactual_properties(
        g0, sigma, prediction, g0, reference_n, omega, beta)
    np.testing.assert_allclose(offset, (prediction-sigma).real.mean(axis=1))
    _, ref_n = occupancy_estimates(g0, omega, beta)
    full, mean, rest = (cases[k][1] for k in ("full", "real_offset_only", "residual_only"))
    np.testing.assert_allclose(full["green"]-g0,
                               mean["green"]-g0 + rest["green"]-g0 + interaction_g)
    np.testing.assert_allclose(full["occupancy_tail"]-ref_n,
                               mean["occupancy_tail"]-ref_n + rest["occupancy_tail"]-ref_n + interaction_n)
    assert np.max(abs(interaction_g)) > 1e-5
    with pytest.raises(AssertionError, match="Dyson consistency"):
        counterfactual_properties(g0, sigma, prediction, g0+.1, reference_n, omega, beta)


def saved_fixture(root):
    folder = root / "energy"
    folder.mkdir(parents=True)
    omega = matsubara_frequencies(20., 32)
    params = np.array([[e, 0., -1., 1., 0., 0., 0., 20.] for e in (-.2, .1)])
    green = 1/(1j*omega - params[:, 0, None])
    sigma = np.zeros_like(green)
    exact = expit(-20*params[:, 0])
    np.savez(folder/"dataset.npz", parameters=params, omega=omega, sigma=sigma,
             g0=green, green=green, occupancy_per_spin=exact, source_indices=[3, 8])
    rows = []
    for seed in (31, 32):
        arrays = dict(parameters=params, omega=omega, reference_sigma=sigma,
                      reference_green=green, reference_occupancy_per_spin=exact)
        for name in ("pca_ridge_8", "grid"):
            prediction = sigma + (.01 if name == "pca_ridge_8" else .02*(seed-30))
            metrics, values = compare_properties(green, prediction, green, exact, omega, 20.)
            rows.append(dict(axis="energy", model_seed=seed, candidate=name,
                             sigma_rmse=float(np.sqrt(np.mean(abs(prediction-sigma)**2))), **metrics))
            arrays[name+"_sigma"] = prediction
            arrays.update({name+"_"+k: v for k, v in values.items()})
        rows.append(dict(axis="energy", model_seed=seed, candidate="failed", status="failed", reason="fixture"))
        np.savez(folder/f"predictions_seed_{seed}.npz", **arrays)
    summary = dict(phase="frozen_reserved_scale", comparisons=rows,
                   dataset_sha256={"energy": sha256_file(folder/"dataset.npz")})
    (root/"summary.json").write_text(json.dumps(summary))


def test_saved_report_all_states_no_models_and_missing_inputs(tmp_path, monkeypatch):
    import joblib
    from dmft_data.learning.models import Baseline

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must not load models, predict or train")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(Baseline, "fit", forbidden)
    monkeypatch.setattr(Baseline, "predict", forbidden)
    root = tmp_path/"reserved"
    saved_fixture(root)
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    out = tmp_path/"report"
    result = report([root], out)
    assert result["status_counts"] == {"ok": 4, "failed": 2}
    assert result["missing"] == []
    assert str(tmp_path) not in json.dumps(result)
    assert all(k.startswith("reserved/") for k in result["artifact_sha256"])
    assert {p: sha256_file(p) for p in before} == before
    with (out/"state_errors.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert len({r["state_id"] for r in rows}) == 2
    assert {r["source_index"] for r in rows} == {"3", "8"}
    with (out/"counterfactual_metrics.csv").open() as handle:
        cf = list(csv.DictReader(handle))
    assert len(cf) == 12 and {r["case"] for r in cf} == {"full", "real_offset_only", "residual_only"}
    assert all(r["status"] == "ok" and r["state_count"] == "2" for r in cf)
    with (out/"counterfactual_states.csv").open() as handle:
        cs = list(csv.DictReader(handle))
    assert len(cs) == 24 and len({r["state_id"] for r in cs}) == 2
    for row in rows:
        r = {k: float(v) for k, v in row.items()
             if k.startswith(("occupancy_", "reference_", "predicted_"))}
        epsilon = float(row["epsilon_d"])
        shift = (.01 if row["candidate"] == "pca_ridge_8"
                 else .02*(int(row["model_seed"])-30))
        # Independent noninteracting result for a shifted level, not a model fit.
        np.testing.assert_allclose(
            r["predicted_occupancy_tail"], expit(-20*(epsilon+shift)), atol=2e-7)
        g = complex(r["predicted_green_iw0_real"], r["predicted_green_iw0_imag"])
        np.testing.assert_allclose(g, 1/(1j*np.pi/20-epsilon-shift))
        np.testing.assert_allclose(r["predicted_sigma_iw0_real"], shift)
        assert r["predicted_sigma_iw0_imag"] == 0
        components = sum(r[k] for k in (
            "occupancy_first_8_contribution", "occupancy_remaining_contribution",
            "occupancy_tail_correction_difference", "reference_tail_minus_exact"))
        np.testing.assert_allclose(components, r["occupancy_error"], atol=1e-15)
        np.testing.assert_allclose(
            r["predicted_occupancy_tail"]-float(row["ed_occupancy_per_spin"]),
            r["occupancy_error"], atol=1e-15)
    with (out/"paired_state_errors.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert (out/"states_00.png").exists()
    with pytest.raises(FileExistsError):
        report([root], out)
    (root/"energy/predictions_seed_32.npz").unlink()
    missing = report([root], tmp_path/"partial")
    assert len(missing["missing"]) == 1
    with (root/"energy/dataset.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        report([root], tmp_path/"invalid")


def test_spectral_link_all_states_and_unsupported_preserved(tmp_path):
    root, spectra = tmp_path/"reserved", tmp_path/"spectra"
    saved_fixture(root)
    prior = json.loads((root/"summary.json").read_text())
    prior.update(protocol_sha256="protocol", model_bundle_sha256="model")
    (root/"summary.json").write_text(json.dumps(prior))
    with np.load(root/"energy/dataset.npz") as d:
        params, exact = d["parameters"], d["occupancy_per_spin"]
    energy = np.linspace(-8, 8, 801)
    eta = .1
    reference = -(1/(energy+.1j-params[:, 0, None])).imag/np.pi
    predicted = -(1/(energy+.1j-params[:, 0, None]-.01)).imag/np.pi
    metrics, _ = spectrum_metrics(predicted, reference, energy, params[:, 7], exact)
    (spectra/"energy").mkdir(parents=True)
    prefix = "seed_31_analytic"
    np.savez(spectra/"energy/spectra_eta_0.1.npz", energy=energy, eta=eta,
             parameters=params, reference_spectrum=reference, exact_occupancy_per_spin=exact,
             **{prefix+"_spectrum": predicted, prefix+"_sigma_infinity": np.full(2, .01),
                prefix+"_sigma_first_moment": np.zeros(2)})
    summary = dict(phase="frozen_impurity_spectra", energy_bound=8, points=801, etas=[eta],
                   dataset_sha256=prior["dataset_sha256"], protocol_sha256="protocol",
                   bundle_sha256="model", matsubara_report_sha256=sha256_file(root/"summary.json"),
                   comparisons=[dict(axis="energy", model_seed=31, candidate="analytic", eta=eta,
                                     status="ok", **metrics),
                                dict(axis="energy", model_seed=31, candidate="pca_ridge_8",
                                     eta=eta, status="unsupported", reason="Matsubara only")])
    (spectra/"summary.json").write_text(json.dumps(summary))
    output = tmp_path/"report"
    result = report([root, spectra], output)
    assert result["missing"] == []
    assert result["status_counts"] == {"ok": 5, "failed": 2, "unsupported": 1}
    with (output/"spectral_state_errors.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    summary["bundle_sha256"] = "different"
    (spectra/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="provenance mismatch"):
        report([root, spectra], tmp_path/"bad_link")

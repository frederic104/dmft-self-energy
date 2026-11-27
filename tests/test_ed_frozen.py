"""Frozen panel provenance, no-fit evaluation and unseen-state checks."""

import json

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from dmft_data.ed_frozen import PANEL, evaluate, freeze, prepare_final
from dmft_data.ed_objective import run_comparison
from dmft_data.ed_workflow import run_study
from dmft_data.learning.models import Baseline
from dmft_data.provenance import sha256_file


def test_frozen_panel_no_refitting_and_target_independence(tmp_path, monkeypatch):
    study, result, panel = (tmp_path / n for n in ("study", "result", "panel"))
    run_study(study, samples=6, frequency_count=16, axes=("energy",))
    run_comparison(study, result, decoder_comparison=True, max_iter=100)
    protocol = freeze(study, result, panel, evaluation_seed=999, samples=7)
    assert protocol["panel"] == list(PANEL)
    bundle = panel / "panel.joblib"
    before = sha256_file(bundle)
    models = joblib.load(bundle)
    assert list(models["energy_seed_31"]) == list(PANEL)

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation must not fit")

    monkeypatch.setattr(Baseline, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    first = evaluate(panel, tmp_path / "first")
    assert len(first["comparisons"]) == 3
    assert all(r["status"] == "ok" for r in first["comparisons"])
    with np.load(tmp_path / "first/energy/dataset.npz") as data:
        assert np.all(data["bath_scale"] == .85)
        assert not np.isclose(data["parameters"][:, 0, None], protocol["cases"][0]["seen_epsilon"],
                              atol=1e-12, rtol=0).any()
    import dmft_data.ed_frozen as module
    original_generator = module.generate_small_dataset

    def changed_targets(path, **kwargs):
        original_generator(path, **kwargs)
        with np.load(path) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays["sigma"] += 10
        arrays["green"] += .1
        arrays["occupancy_per_spin"] += .1
        np.savez(path, **arrays)

    monkeypatch.setattr(module, "generate_small_dataset", changed_targets)
    second = evaluate(panel, tmp_path / "second")
    with np.load(tmp_path / "first/energy/predictions_seed_31.npz") as a, np.load(
            tmp_path / "second/energy/predictions_seed_31.npz") as b:
        for name in PANEL:
            np.testing.assert_array_equal(a[f"{name}_sigma"], b[f"{name}_sigma"])
    assert first["comparisons"][0]["sigma_rmse"] != second["comparisons"][0]["sigma_rmse"]
    assert sha256_file(bundle) == before
    with pytest.raises(FileExistsError):
        evaluate(panel, tmp_path / "first")
    # A new seed is insufficient if it reproduces a development epsilon value.
    p = json.loads((panel / "protocol.json").read_text())
    p["cases"][0]["seen_epsilon"].append(float(np.random.default_rng(999).uniform(-1, .2)))
    (panel / "protocol.json").write_text(json.dumps(p))
    with pytest.raises(ValueError, match="overlaps"):
        evaluate(panel, tmp_path / "overlap")
    with bundle.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        evaluate(panel, tmp_path / "tampered")


def test_closed_panel_preparation_and_reserved_evaluation(tmp_path, monkeypatch):
    import dmft_data.ed_frozen as module
    study, result, panel = (tmp_path / n for n in ("study", "result", "panel"))
    run_study(study, samples=6, frequency_count=16, axes=("energy",))
    run_comparison(study, result, decoder_comparison=True, max_iter=100)
    freeze(study, result, panel)
    source = json.loads((result / "energy/seed_31/metrics.json").read_text())
    with np.load(study / "energy/dataset.npz") as data:
        training_sigma = data["sigma"][source["splits"]["train"]].copy()
        reserved_sigma = data["sigma"][source["splits"]["test"]].copy()
    original_refine = module.refine_model
    calls = []

    def train_only(model, x, sigma, *args, **kwargs):
        np.testing.assert_array_equal(sigma, training_sigma)
        calls.append(kwargs["weight"])
        return original_refine(model, x, sigma, *args, **kwargs)

    monkeypatch.setattr(module, "refine_model", train_only)
    final = tmp_path / "final"
    protocol = prepare_final(panel, study, result, final)
    assert calls == [1., 0., 1., 0.]
    assert protocol["search_status"] == "closed"
    assert protocol["model_seeds"] == [31, 32, 33]
    assert "evaluation_seed" not in protocol
    old = joblib.load(panel / "panel.joblib")
    new = joblib.load(final / "panel.joblib")
    for name in PANEL:
        for a, b in zip(old["energy_seed_31"][name].model.coefs_ if name != PANEL[0] else [old["energy_seed_31"][name].model.coef_],
                        new["energy_seed_31"][name].model.coefs_ if name != PANEL[0] else [new["energy_seed_31"][name].model.coef_]):
            np.testing.assert_array_equal(a, b)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reserved evaluation may neither solve new ED states nor fit")

    monkeypatch.setattr(module, "generate_small_dataset", forbidden)
    monkeypatch.setattr(module, "refine_model", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    report = evaluate(final, tmp_path / "evaluated")
    assert report["phase"] == "frozen_reserved_scale"
    assert len(report["comparisons"]) == 9
    assert all(r["status"] == "ok" for r in report["comparisons"])
    assert report["seed_summary"][0]["sigma_rmse_std"] == pytest.approx(0, abs=1e-14)
    assert all(r["seed_count"] == 3 for r in report["seed_summary"])
    with np.load(tmp_path / "evaluated/energy/dataset.npz") as data:
        np.testing.assert_array_equal(data["sigma"], reserved_sigma)
        assert np.all(data["bath_scale"] == 1.15)
    with (final / "energy/reserved.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="Reserved dataset checksum"):
        evaluate(final, tmp_path / "bad")

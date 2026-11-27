"""Frozen panel provenance, no-fit evaluation and unseen-state checks."""

import json

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from dmft_data.ed_frozen import PANEL, evaluate, freeze
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

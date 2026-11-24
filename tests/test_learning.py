"""Scientific workflow checks using small, local TRIQS-shaped fixtures."""

import copy
import json
import subprocess
import sys

import h5py
import joblib
import numpy as np
import pytest
import yaml

from dmft_data.hdf import read_quantity
from dmft_data.learning.data import catalog, load_json, prepare, write_json
from dmft_data.learning.experiment import causality_metrics, run_experiment
from dmft_data.learning.models import encode
from dmft_data.learning.splits import blocked_partition
from dmft_data.provenance import sha256_file


def write_run(path, parameter, beta=40.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    omega = 1j * (2 * np.arange(-8, 8) + 1) * np.pi / beta
    with h5py.File(path, "w") as handle:
        for quantity, values in (
            ("G0-1", 1 / (omega + parameter)),
            ("Sigma_iw", parameter + parameter**2 / (omega - 0.3)),
            ("Sigma-5", parameter + parameter**2 / (omega - 0.3)),
            ("Sigma-15", parameter + 3 * parameter**2 / (omega - 0.3)),
        ):
            parent = handle.create_group("dmft_output/" + quantity)
            parent.create_dataset("block_names/0", data="up_0")
            for block in ("down_0", "up_0"):
                group = parent.create_group(block)
                raw = np.stack([values.real, values.imag], axis=-1)[:, None, None, :]
                data = group.create_dataset("data", data=raw)
                data.attrs["__complex__"] = "1"
                mesh = group.create_group("mesh")
                mesh.create_dataset("size", data=len(omega))
                mesh.create_dataset("positive_freq_only", data=0)
                mesh.create_dataset("domain/beta", data=beta)
                mesh.create_dataset("domain/statistic", data="Fermion")


def setup_experiment(tmp_path):
    materials = ["SVO", "SNO", "CVO", "BVO"]
    for i, material in enumerate(materials):
        for replica in (0, 1):
            write_run(
                tmp_path / "raw" / material / f"run{replica}.h5", 1 + i + replica * 0.01
            )
    review = tmp_path / "review.json"
    catalog(tmp_path / "raw", review)
    return {
        "source": "raw",
        "review": "review.json",
        "prepared": "prepared",
        "experiment": "experiment",
        "positive_frequency_count": 4,
        "include_all": True,
        "grouping": "material_beta_candidate",
        "task": "exploratory",
        "selection": {
            "input_path": "/dmft_output/G0-1",
            "target_path": "/dmft_output/Sigma_iw",
            "input_semantics": "unresolved",
        },
        "split": {"dimension": "material", "validation": ["CVO"], "test": ["BVO"]},
        "models": [
            {"name": "mean"},
            {"name": "nearest"},
            {"name": "pca_ridge", "latent_dimension": 4},
            {"name": "mlp", "hidden_layers": [4], "max_iter": 5},
        ],
        "seeds": [3],
    }


def test_selected_path_complex_mesh_and_metadata_only_blocks(tmp_path):
    path = tmp_path / "BVO.h5"
    write_run(path, 2)
    with h5py.File(path) as handle:
        final = read_quantity(handle, "/dmft_output/Sigma_iw", 4)
        earlier = read_quantity(handle, "/dmft_output/Sigma-5", 4)
        last = read_quantity(handle, "/dmft_output/Sigma-15", 4)
    assert final.blocks == ["down_0", "up_0"]
    assert final.values.shape == (2, 1, 1, 4)
    assert np.array_equal(final.values, earlier.values)
    assert not np.array_equal(final.values, last.values)
    assert np.all(final.mesh.imag > 0)


def test_end_to_end_preparation_training_reload_and_reproducibility(tmp_path):
    config = setup_experiment(tmp_path)
    manifest = prepare(config, tmp_path)
    report = run_experiment(config, tmp_path)
    assert manifest["sample_count"] == 8
    assert manifest["group_count"] == 4
    assert all(r["physical_state_id"] is None for r in manifest["records"])
    assert all(r["convergence_status"] == "not_assessed" for r in manifest["records"])
    assert report["status"] == "completed"
    assert report["task"] == "exploratory"
    splits = load_json(tmp_path / "experiment/splits.json")
    assert len(splits["train"]["groups"]) == 2
    assert len(splits["test"]["indices"]) == 2
    # Preprocessing sees only the two training group means, never held-out values.
    model = joblib.load(tmp_path / "experiment/candidate_0_seed_3.joblib")
    with np.load(tmp_path / "prepared/arrays.npz") as arrays:
        x = np.concatenate([encode(arrays["inputs"]), arrays["meshes"].imag], axis=1)
        assert np.allclose(
            model.x_scaler.mean_, x[splits["train"]["indices"]].mean(axis=0)
        )
    repeated = {**config, "experiment": "repeat"}
    second = run_experiment(repeated, tmp_path)
    assert second["test"] == report["test"]
    assert report["candidates"][2]["seeds"][0]["actual_latent_dimension"] == 1
    assert (tmp_path / "experiment/test_predictions_seed_3.npz").is_file()


def test_primary_needs_hash_and_target_scoped_evidence(tmp_path):
    config = setup_experiment(tmp_path)
    config["task"] = "primary"
    rejected = prepare(config, tmp_path)
    assert rejected["sample_count"] == 0
    assert len(rejected["excluded"]) == 8
    review = load_json(tmp_path / "review.json")
    for run in review["runs"]:
        run["physical_state_id"] = run["material"] + "-reviewed"
        run["metadata_evidence"] = "synthetic Hamiltonian fixture"
        run["input_semantics"] = "dmft_weiss"
        run["target_semantics"] = "impurity_self_energy"
        run["convergence"] = {
            "status": "verified_converged",
            "evidence_source": "synthetic test only",
            "source_sha256": sha256_file(
                tmp_path / "raw" / run["relative_source_path"]
            ),
            "target_path": "/dmft_output/Sigma_iw",
        }
    write_json(tmp_path / "review.json", review)
    accepted = prepare({**config, "prepared": "primary"}, tmp_path)
    assert accepted["sample_count"] == 8
    # A different selected target never inherits final-array convergence evidence.
    bad = copy.deepcopy(config)
    bad["prepared"] = "wrong-target"
    bad["selection"]["target_path"] = "/dmft_output/Sigma-15"
    assert prepare(bad, tmp_path)["sample_count"] == 0


def test_no_unknown_group_or_silent_missing_path_and_no_overwrite(tmp_path):
    config = setup_experiment(tmp_path)
    config.pop("grouping")
    report = prepare(config, tmp_path)
    assert report["sample_count"] == 0
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path)
    bad = {
        **config,
        "prepared": "bad",
        "grouping": "material_beta_candidate",
        "selection": {"input_path": "/missing", "target_path": "/missing-target"},
    }
    report = prepare(bad, tmp_path)
    assert report["sample_count"] == 0
    assert "unavailable" in report["excluded"][0]["reason"]


def test_replica_leakage_and_unknown_split_metadata_rejected():
    records = [
        {"group_id": "same", "material": m, "relative_source_path": m}
        for m in ["SVO", "CVO", "BVO"]
    ]
    with pytest.raises(ValueError, match="crosses"):
        blocked_partition(records, {"validation": ["CVO"], "test": ["BVO"]})
    records[0]["material"] = None
    with pytest.raises(ValueError, match="Missing"):
        blocked_partition(records, {"validation": ["CVO"], "test": ["BVO"]})


def test_numeric_beta_holdout_labels_match_yaml_integer_values():
    records = [
        {"group_id": str(i), "beta": beta, "relative_source_path": str(i)}
        for i, beta in enumerate([20.0, 40.0, 60.0])
    ]
    result = blocked_partition(
        records, {"dimension": "beta", "validation": [40], "test": [60]}
    )
    assert result == {"train": [0], "validation": [1], "test": [2]}


def test_changed_prepared_arrays_rejected(tmp_path):
    config = setup_experiment(tmp_path)
    prepare(config, tmp_path)
    with (tmp_path / "prepared/arrays.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_experiment(config, tmp_path)


def test_exact_copies_cannot_cross_partitions(tmp_path):
    import shutil

    config = setup_experiment(tmp_path)
    shutil.copyfile(tmp_path / "raw/CVO/run0.h5", tmp_path / "raw/BVO/run0.h5")
    prepare(config, tmp_path)
    with pytest.raises(ValueError, match="payload replica"):
        run_experiment(config, tmp_path)


def test_test_values_cannot_change_validation_selection(tmp_path):
    config = setup_experiment(tmp_path)
    prepare(config, tmp_path)
    original = run_experiment(config, tmp_path)
    for replica in (0, 1):
        write_run(tmp_path / "raw/BVO" / f"run{replica}.h5", 20 + replica * 0.01)
    changed = {**config, "prepared": "changed_prepared", "experiment": "changed_test"}
    prepare(changed, tmp_path)
    repeated = run_experiment(changed, tmp_path)
    assert original["selected_candidate"] == repeated["selected_candidate"]
    assert original["candidates"] == repeated["candidates"]
    assert (
        original["test"][0]["errors"]["rmse"] != repeated["test"][0]["errors"]["rmse"]
    )


def test_matrix_causality_sign_check():
    causal = np.zeros((1, 1, 2, 2, 4), dtype=complex)
    causal[:, :, 0, 0] = -1j
    causal[:, :, 1, 1] = -2j
    assert causality_metrics(causal)["positive_imaginary_eigenvalue_fraction"] == 0
    causal[:, :, 0, 1] = 4j
    causal[:, :, 1, 0] = 4j
    assert causality_metrics(causal)["positive_imaginary_eigenvalue_fraction"] > 0


def test_flat_schema_inspector_finds_real_stored_iterations(tmp_path):
    from dmft_data.triqs_inspection import inspect_triqs_file

    path = tmp_path / "schema.h5"
    write_run(path, 2)
    report = inspect_triqs_file(path, tmp_path)
    assert report["status"] == "success"
    assert report["stored_iterations_by_quantity"]["Sigma"] == ["5", "15"]
    assert report["quantities"]["Sigma"]["channels"] == ["down_0", "up_0"]
    assert report["convergence"]["status"] == "not_assessed"


def test_cli_prepare_and_train_from_external_cwd(tmp_path):
    config = setup_experiment(tmp_path)
    config["repository_root"] = str(tmp_path)
    config["models"] = [{"name": "mean"}]
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    for command in ["prepare", "train", "summary"]:
        result = subprocess.run(
            [sys.executable, "-m", "dmft_data", command, "--config", str(config_path)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert (
        json.loads((tmp_path / "experiment/metrics.json").read_text())["task"]
        == "exploratory"
    )

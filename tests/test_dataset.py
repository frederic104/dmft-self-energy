"""Tests for state-aware research-dataset construction."""

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from dmft_data.dataset import build_research_datasets


def _write_normalized_run(
    path: Path,
    state_id: str,
    run_id: str,
    material: str,
    temperature: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = np.array([1j, 3j, 5j])
    iw = np.array([1j, 3j, 5j])
    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = run_id
        handle.attrs["physical_state_id"] = state_id
        handle.attrs["run_id_explicit"] = True
        handle.attrs["physical_state_id_explicit"] = True
        handle.attrs["source_sha256"] = "a" * 64
        handle.attrs["material"] = material
        handle.attrs["temperature_K"] = temperature
        arrays = handle.create_group("run").create_group("arrays")
        sigma = arrays.create_dataset(
            "array_0000",
            data=np.ones((2, 3), dtype=np.complex128),
        )
        sigma.attrs["role"] = "impurity_self_energy"
        sigma.attrs["source_path"] = "/dmft/iteration_4/Sigma"
        sigma.attrs["iteration"] = "4"
        sigma.attrs["frequency_axis"] = 1
        sigma.attrs["spin_labels"] = ["up", "down"]
        sigma.attrs["orbital_labels"] = ["dxz", "dyz"]
        mesh_dataset = arrays.create_dataset("mesh_0000", data=mesh)
        mesh_dataset.attrs["source_path"] = "/iw"

        g0 = arrays.create_dataset(
            "array_0001",
            data=np.zeros((2, 3), dtype=np.complex128),
        )
        g0.attrs["role"] = "dft_derived_g0"
        g0.attrs["source_path"] = "/dmft/iteration_4/G0"
        g0.attrs["iteration"] = "4"
        g0.attrs["frequency_axis"] = 1
        g0_mesh = arrays.create_dataset("mesh_0001", data=iw)
        g0_mesh.attrs["source_path"] = "/iw"

        warmup = arrays.create_dataset(
            "array_0002",
            data=np.full((2, 3), 9j, dtype=np.complex128),
        )
        warmup.attrs["role"] = "dmft_weiss"
        warmup.attrs["source_path"] = "/dmft/warmup/Weiss"
        warmup.attrs["iteration"] = "warmup"
        warmup.attrs["warmup"] = True
        warmup.attrs["frequency_axis"] = 1
        warmup_mesh = arrays.create_dataset("mesh_0002", data=iw)
        warmup_mesh.attrs["source_path"] = "/iw"


def _config(tmp_path: Path, minimum_states: int = 3) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    config = root / "configs" / "dataset.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test"},
                "paths": {
                    "processed_dir": "data/processed",
                    "report_dir": "reports",
                    "raw_dir": "data/raw/hdf5",
                },
                "dataset": {
                    "physical_state_metadata": "data/metadata/physical_states.json",
                    "convergence_verification": "data/metadata/convergence_verification.json",
                    "splits": {"minimum_physical_states": minimum_states},
                    "frequency_windows": [
                        {"name": "low_frequency", "count": 2},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return root, config


def _verification(*run_ids: str) -> str:
    return json.dumps(
        {
            "runs": {
                run_id: {
                    "status": "verified_converged",
                    "evidence_source": "synthetic test only",
                    "source_sha256": "a" * 64,
                    "verified_source_paths": [
                        "/dmft/iteration_4/Sigma",
                        "/dmft/iteration_4/G0",
                    ],
                }
                for run_id in run_ids
            }
        }
    )


def test_builds_converged_primary_and_blocked_splits(tmp_path: Path) -> None:
    root, config = _config(tmp_path)
    normalized = root / "data" / "processed" / "arrays"
    for index, (state, material) in enumerate(
        [("state-A", "SVO"), ("state-B", "CVO"), ("state-C", "BVO")]
    ):
        _write_normalized_run(
            normalized / f"run-{index}.h5",
            state,
            f"run-{index}",
            material,
            "300",
        )

    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "convergence_verification.json").write_text(
        _verification("run-0", "run-1", "run-2"),
        encoding="utf-8",
    )

    report = build_research_datasets(config, root)

    assert report["status"] == "passed"
    assert report["blocked_splits"]["independent_physical_state_count"] == 3
    assert set(report["blocked_splits"]["independent_physical_state_ids"]) == {
        "state-A",
        "state-B",
        "state-C",
    }
    assert report["baseline"]["record_count"] == 3
    assert report["physics_constrained"]["record_count"] == 6
    assert report["compatibility"]["record_count"] == 9
    assert report["auxiliary"]["record_count"] == 3
    assert len(report["derived"]) == 3

    split_path = (
        root / "data" / "processed" / "research_datasets" / "blocked_splits.json"
    )
    splits = __import__("json").loads(split_path.read_text())
    assert splits["splits"]["material"]["status"] == "created"
    folds = splits["splits"]["material"]["folds"]
    assert {fold["held_out_group"] for fold in folds} == {"SVO", "CVO", "BVO"}
    assert all(len(fold["train_state_ids"]) == 2 for fold in folds)


def test_insufficient_states_does_not_create_misleading_splits(tmp_path: Path) -> None:
    root, config = _config(tmp_path)
    normalized = root / "data" / "processed" / "arrays"
    _write_normalized_run(normalized / "run-a.h5", "state-A", "run-a", "SVO", "300")
    _write_normalized_run(normalized / "run-b.h5", "state-B", "run-b", "CVO", "300")

    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "convergence_verification.json").write_text(
        _verification("run-a", "run-b"),
        encoding="utf-8",
    )

    report = build_research_datasets(config, root)

    assert report["status"] == "insufficient_physical_states"
    assert report["blocked_splits"]["status"] == "insufficient_physical_states"
    assert report["blocked_splits"]["splits"] == {}


def test_derived_window_selects_first_positive_matsubara_frequencies(
    tmp_path: Path,
) -> None:
    root, config = _config(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["dataset"]["frequency_windows"] = [
        {
            "name": "positive_low_frequency",
            "count": 2,
            "positive_frequency": True,
        }
    ]
    config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    normalized = root / "data" / "processed" / "arrays"
    _write_normalized_run(normalized / "run-a.h5", "state-A", "run-a", "SVO", "300")

    metadata_dir = root / "data" / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "convergence_verification.json").write_text(
        _verification("run-a"),
        encoding="utf-8",
    )

    report = build_research_datasets(config, root)

    assert len(report["derived"]) == 3
    derived_path = root / "data" / "processed" / "research_datasets" / "derived"
    output = derived_path / "deep_dmft_positive_low_frequency.h5"
    with h5py.File(output, "r") as handle:
        group = handle["runs"]["record_000000"]
        mesh = group["matsubara_mesh"][()]
        values = group["values"][()]
        assert np.all(np.imag(mesh) > 0)
        assert mesh.shape == (2,)
        assert values.shape == (2, 2)
        assert np.array_equal(group.attrs["window_source_indices"], np.array([0, 1]))


def test_global_convergence_boolean_cannot_certify_a_collection(tmp_path: Path) -> None:
    root, config = _config(tmp_path)
    normalized = root / "data/processed/arrays"
    _write_normalized_run(normalized / "run.h5", "state-A", "run-A", "SVO", "300")
    metadata_dir = root / "data/metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "convergence_verification.json").write_text(
        '{"convergence_verified": true, "converged_iterations": ["4"]}',
        encoding="utf-8",
    )
    report = build_research_datasets(config, root)
    assert report["baseline"]["record_count"] == 0
    assert report["physics_constrained"]["record_count"] == 0
    scoped = json.loads(_verification("run-A"))
    scoped["runs"]["run-A"]["evidence_source"] = None
    (metadata_dir / "convergence_verification.json").write_text(
        json.dumps(scoped), encoding="utf-8"
    )
    assert build_research_datasets(config, root)["baseline"]["record_count"] == 0

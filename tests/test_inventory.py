"""Unit tests for the non-destructive HDF5 inventory."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from dmft_data.inventory import (
    candidate_categories,
    inspect_hdf5_file,
    run_inventory,
)


def test_candidate_categories_are_name_only() -> None:
    categories = candidate_categories("/dmft/sigma_omega")
    assert "candidate_self_energy" in categories
    assert "candidate_frequency_mesh" in categories
    assert "candidate_green_function" not in categories
    assert candidate_categories("/dos") == ""


def test_inspect_hdf5_file_records_schema_and_attributes(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    nested = source_root / "SVO"
    nested.mkdir(parents=True)
    path = nested / "run.h5"

    with h5py.File(path, "w") as handle:
        handle.attrs["source"] = "synthetic"
        handle.attrs["calculation_id"] = 42
        group = handle.create_group("dmft")
        group.attrs["iteration_note"] = "unknown"
        group.create_dataset(
            "Sigma_matsubara", data=np.ones((3, 2), dtype=np.complex128)
        )
        group.create_dataset("G0", data=np.zeros((3,), dtype=np.float64))
        group.create_dataset("freq_mesh", data=np.array([-1.0, 0.0, 1.0]))

    file_row, group_rows, dataset_rows, failures = inspect_hdf5_file(path, source_root)

    assert file_row["status"] == "success"
    assert file_row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(file_row["file_attributes"])["calculation_id"] == 42
    assert {row["group_path"] for row in group_rows} == {"/", "/dmft"}
    assert len(dataset_rows) == 3
    assert failures == []

    datasets = {row["dataset_path"]: row for row in dataset_rows}
    assert datasets["/dmft/Sigma_matsubara"]["shape"] == "[3, 2]"
    assert (
        "candidate_self_energy"
        in datasets["/dmft/Sigma_matsubara"]["candidate_categories"]
    )
    assert (
        "candidate_frequency_mesh"
        in datasets["/dmft/freq_mesh"]["candidate_categories"]
    )
    assert "candidate_green_function" in datasets["/dmft/G0"]["candidate_categories"]


def test_run_inventory_writes_csv_and_markdown(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    raw_dir = repository_root / "data" / "raw" / "hdf5"
    raw_dir.mkdir(parents=True)
    with h5py.File(raw_dir / "sample.hdf5", "w") as handle:
        handle.attrs["format"] = "test"
        handle.create_dataset("Gimp", data=np.arange(4))

    config_path = repository_root / "configs" / "dataset.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "paths": {
            "raw_dir": "data/raw/hdf5",
            "metadata_dir": "data/metadata",
            "report_dir": "reports",
        },
        "dataset": {
            "file_extensions": [".h5", ".hdf5"],
            "recursive": True,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_inventory(config_path, repository_root)

    assert len(result["files"]) == 1
    assert len(result["datasets"]) == 1
    metadata_dir = repository_root / "data" / "metadata"
    report_dir = repository_root / "reports"

    with (metadata_dir / "hdf5_files.csv").open(newline="", encoding="utf-8") as handle:
        file_rows = list(csv.DictReader(handle))
    assert file_rows[0]["relative_path"] == "sample.hdf5"
    assert len(file_rows[0]["sha256"]) == 64
    assert (metadata_dir / "hdf5_datasets.csv").is_file()
    assert (metadata_dir / "hdf5_groups.csv").is_file()
    assert (metadata_dir / "hdf5_failures.csv").is_file()
    assert (metadata_dir / "hdf5_inventory_metadata.json").is_file()
    assert (report_dir / "hdf5_inventory.md").is_file()


def test_corrupt_hdf5_is_recorded_not_silently_skipped(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    path = source_root / "broken.h5"
    path.write_text("not an hdf5 file", encoding="utf-8")

    file_row, group_rows, dataset_rows, failures = inspect_hdf5_file(path, source_root)

    assert file_row["status"] == "failed"
    assert file_row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert group_rows == []
    assert dataset_rows == []
    assert len(failures) == 1
    assert failures[0]["relative_path"] == "broken.h5"


def test_missing_configured_directory_is_reported(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    config_path = repository_root / "configs" / "dataset.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "paths": {
            "raw_dir": "missing",
            "metadata_dir": "data/metadata",
            "report_dir": "reports",
        },
        "dataset": {"file_extensions": [".h5"], "recursive": True},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_inventory(config_path, repository_root)

    assert len(result["files"]) == 1
    assert result["files"][0]["status"] == "failed"
    assert len(result["failures"]) == 1

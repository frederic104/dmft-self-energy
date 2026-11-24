"""Tests for the laptop-side HDF5 reporting bundle."""

import csv
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_collect_laptop_results.py"
SPEC = importlib.util.spec_from_file_location("collect_laptop_results", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
collect_laptop_results = MODULE.collect_laptop_results


def test_collect_laptop_results_writes_transfer_bundle(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    hdf5_path = source / "sample.h5"
    mesh = np.array([1j, 3j, 5j])
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["format"] = "laptop-test"
        handle.create_dataset("iw", data=mesh)
        handle.create_dataset(
            "dmft/iteration_1/Sigma",
            data=np.ones((2, 3), dtype=np.complex128),
        )

    output = tmp_path / "laptop-results"
    aggregate, zip_path = collect_laptop_results(source, output)

    assert aggregate["status"] == "passed"
    assert aggregate["source_count"] == 1
    assert aggregate["files"][0]["sha256"] == hashlib.sha256(
        hdf5_path.read_bytes()
    ).hexdigest()

    csv_path = output / "laptop_collection.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["relative_source_path"] == "sample.h5"

    aggregate_path = output / "laptop_collection.json"
    assert json.loads(aggregate_path.read_text(encoding="utf-8"))["source_count"] == 1
    aggregate_text = aggregate_path.read_text(encoding="utf-8")
    assert str(source.resolve()) not in aggregate_text
    assert "<LOCAL_SOURCE_ROOT>" in aggregate_text
    assert zipfile.is_zipfile(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        per_file_text = archive.read("per_file/sample.json").decode("utf-8")
    assert str(source.resolve()) not in per_file_text

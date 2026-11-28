"""Tests for read-only TRIQS-style HDF5 inspection."""

import json
from pathlib import Path

import h5py
import numpy as np

from dmft_data.triqs_inspection import inspect_triqs_file
from dmft_data.provenance import redact_source_root


def test_triqs_iterations_metadata_and_stored_iterations_are_separate(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    path = source_root / "triqs.h5"

    with h5py.File(path, "w") as handle:
        handle.attrs["iterations"] = 55
        for quantity in ("G", "G0", "Sigma"):
            quantity_group = handle.create_group(quantity)
            for iteration in (1, 2):
                iteration_group = quantity_group.create_group(str(iteration))
                for block in ("up_0", "down_0"):
                    block_group = iteration_group.create_group(block)
                    block_group.attrs["MeshImFreq"] = True
                    block_group.create_dataset(
                        "data",
                        data=(
                            np.ones((3,), dtype=np.complex128)
                            if iteration == 1
                            else np.full((3,), 2.0, dtype=np.complex128)
                        ),
                    )
                    block_group.create_dataset(
                        "MeshImFreq",
                        data=np.array([1j, 3j, 5j]),
                    )

    report = inspect_triqs_file(path, source_root)

    assert report["status"] == "success"
    assert report["sha256"]
    assert str(source_root) not in json.dumps(report)
    assert report["source_path"].startswith("<LOCAL_SOURCE_ROOT>")
    assert report["iteration_evidence"] == [
        {
            "source_path": "/",
            "attribute": "iterations",
            "value": 55,
        }
    ]
    for quantity in ("G", "G0", "Sigma"):
        assert report["quantities"][quantity]["stored_iterations"] == ["1", "2"]
        assert set(report["quantities"][quantity]["channels"]) == {
            "up_0",
            "down_0",
        }
    assert len(report["iteration_diagnostics"]) == 6
    assert report["convergence"]["status"] == "not_assessed"
    assert report["convergence"]["last_iteration_is_converged"] is False


def test_report_is_json_serializable(tmp_path: Path) -> None:
    path = tmp_path / "single.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "triqs-test"

    report = inspect_triqs_file(path, tmp_path)

    assert json.loads(json.dumps(report))["sha256"] == report["sha256"]


def test_redaction_keeps_hdf5_paths_and_handles_slash_variants():
    original = {"path": "C:/private/data/file.h5",
                "nested": [{"error": r"failed C:\private\data\file.h5"}],
                "hdf5_path": "/dmft_output/Sigma-1", "sha256": "unchanged"}
    redacted = redact_source_root(original, r"C:\private\data")
    assert "private" not in json.dumps(redacted)
    assert redacted["hdf5_path"] == original["hdf5_path"]
    assert redacted["sha256"] == "unchanged"
    assert original["path"] == "C:/private/data/file.h5"

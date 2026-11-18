"""Numerical validation and physical-convergence separation tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dmft_data.extraction import UnsupportedSchemaError, read_hdf5_file
from dmft_data.validation import validate_extraction


def test_frequency_mesh_shape_validation(tmp_path: Path) -> None:
    path = tmp_path / "mesh.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iw", data=np.array([1j, 3j, 5j]))
        handle.create_dataset(
            "dmft/Sigma",
            data=np.ones((2, 2), dtype=np.complex128),
        )

    with pytest.raises(UnsupportedSchemaError):
        read_hdf5_file(path)


def test_missing_datasets_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "empty"

    extraction = read_hdf5_file(path)
    report = validate_extraction(extraction)

    assert report.status == "failed"
    assert any(issue.code == "missing_datasets" for issue in report.issues)
    assert report.physical_convergence["status"] == "not_assessed"
    assert report.physical_convergence["last_iteration_is_converged"] is False


def test_iteration_diagnostics_are_calculated(tmp_path: Path) -> None:
    path = tmp_path / "iterations.h5"
    iw = np.array([1j, 3j, 5j])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iw", data=iw)
        first = handle.create_dataset(
            "dmft/iteration_1/Sigma",
            data=np.ones((2, 3), dtype=np.complex128),
        )
        first.attrs["frequency_axis"] = 1
        second = handle.create_dataset(
            "dmft/iteration_2/Sigma",
            data=2.0 * np.ones((2, 3), dtype=np.complex128),
        )
        second.attrs["frequency_axis"] = 1

    report = validate_extraction(read_hdf5_file(path))

    assert report.status == "passed"
    assert len(report.iteration_diagnostics) == 1
    diagnostic = report.iteration_diagnostics[0]
    assert diagnostic["from_iteration"] == "1"
    assert diagnostic["to_iteration"] == "2"
    assert diagnostic["max_abs"] == pytest.approx(1.0)

"""Tests for run-level HDF5 classification."""

from pathlib import Path

import h5py
import numpy as np

from dmft_data.run_diagnostics import classify_hdf5_run, classify_hdf5_runs


def _write_run(
    path: Path,
    mismatch: bool = False,
    complex_marker: int | str = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.zeros((2050, 1, 1, 2), dtype=np.float64)
    raw[..., 0] = 1.0
    raw[..., 1] = 2.0
    with h5py.File(path, "w") as handle:
        dmft = handle.create_group("dmft_output")
        for block_name in ("down_0", "up_0"):
            for quantity in ("G", "Sigma"):
                final = dmft.require_group(f"{quantity}_iw")
                final_block = final.create_group(block_name)
                final_data = final_block.create_dataset("data", data=raw)
                final_data.attrs["__complex__"] = complex_marker

                iteration = dmft.require_group(f"{quantity}-15")
                iteration_block = iteration.create_group(block_name)
                iteration_raw = raw.copy()
                if mismatch:
                    iteration_raw[..., 0] += 1.0
                iteration_data = iteration_block.create_dataset(
                    "data",
                    data=iteration_raw,
                )
                iteration_data.attrs["__complex__"] = complex_marker

        mesh = dmft.create_group("G_iw/down_0/mesh")
        domain = mesh.create_group("domain")
        domain.create_dataset("beta", data=np.float64(40.0))
        dmft.create_dataset("iterations", data=np.int64(15))


def test_classifies_equivalent_and_mismatched_runs(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    equivalent = root / "SNO" / "SNO_beta40_nwarmup150000.h5"
    mismatch = root / "BVO" / "BVO_beta40_nwarmup100000.h5"
    _write_run(equivalent)
    _write_run(mismatch, mismatch=True)

    equivalent_row = classify_hdf5_run(equivalent, root)
    mismatch_row = classify_hdf5_run(mismatch, root)

    assert equivalent_row["run_status"] == "extracted_final_equivalent"
    assert equivalent_row["primary_target_candidate"] is True
    assert equivalent_row["mesh_status"] == "unresolved"
    assert equivalent_row["convergence_status"] == "not_assessed"
    assert mismatch_row["run_status"] == "final_iteration_mismatch"
    assert mismatch_row["primary_target_candidate"] is False


def test_writes_run_diagnostics_reports(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "SNO" / "beta-variation" / "SNO_beta20_nwarmup150000.h5"
    _write_run(path)

    summary = classify_hdf5_runs(root, tmp_path / "reports")

    assert summary["file_count"] == 1
    assert summary["files_with_verified_mesh"] == 0
    assert summary["convergence_status"] == "not_assessed"
    assert (tmp_path / "reports" / "run_diagnostics.json").is_file()
    assert (tmp_path / "reports" / "run_diagnostics.csv").is_file()


def test_classifies_string_complex_marker(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "SVO" / "SVO_beta40.h5"
    _write_run(path, complex_marker="1")

    row = classify_hdf5_run(path, root)

    assert row["run_status"] == "extracted_final_equivalent"

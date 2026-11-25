"""Tests for storage-only run-level HDF5 diagnostics."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dmft_data.run_diagnostics import classify_hdf5_run, classify_hdf5_runs


def _raw(value: float) -> np.ndarray:
    data = np.zeros((8, 1, 1, 2), dtype=np.float64)
    data[..., 0] = value
    data[..., 1] = 2.0
    return data


def _write_run(path: Path, final_source: int | None) -> None:
    """Write G/Sigma iterations; ``None`` makes final match no iteration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dmft = handle.create_group("dmft_output")
        for quantity in ("G", "Sigma"):
            for iteration, value in ((3, 3.0), (15, 15.0)):
                data = dmft.create_dataset(
                    f"{quantity}-{iteration}/down_0/data", data=_raw(value)
                )
                data.attrs["__complex__"] = "1"
            final_value = 99.0 if final_source is None else float(final_source)
            final = dmft.create_dataset(
                f"{quantity}_iw/down_0/data", data=_raw(final_value)
            )
            final.attrs["__complex__"] = "1"
            dmft.create_dataset(f"{quantity}_iw/block_names/0", data="down_0")
        mesh = dmft.create_group("G_iw/down_0/mesh")
        mesh.create_dataset("positive_freq_only", data=np.int32(0))
        mesh.create_group("domain").create_dataset("beta", data=np.float64(40.0))


def test_final_equals_last_iteration_with_string_complex_marker(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "SNO" / "SNO_beta40_nwarmup150000.h5"
    _write_run(path, final_source=15)

    row = classify_hdf5_run(path, root, low_frequency_count=2)

    assert row["run_status"] == "final_matches_last_iteration"
    assert row["convergence_status"] == "not_assessed"
    assert row["primary_target_candidate"] is False
    block = row["G_iw_storage"]["blocks"][0]
    assert block["numerical_last_iteration"] == 15
    assert block["closest_stored_iteration"] == 15
    assert block["exact_match_iterations"] == [15]
    assert [entry["iteration"] for entry in block["iteration_comparisons"]] == [3, 15]
    assert block["closest_stored_iteration_metrics"]["max_abs"] == 0.0
    assert block["closest_stored_iteration_metrics"]["relative_rms"] == 0.0
    assert block["low_frequency_selection"]["requested_positive_frequency_count"] == 2


def test_final_equals_earlier_iteration(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "BVO" / "BVO_beta40_nwarmup100000.h5"
    _write_run(path, final_source=3)

    row = classify_hdf5_run(path, root)

    assert row["run_status"] == "final_matches_earlier_iteration"
    block = row["Sigma_iw_storage"]["blocks"][0]
    assert block["closest_stored_iteration"] == 3
    assert block["exact_match_iterations"] == [3]
    assert block["last_iteration_comparison"]["iteration"] == 15
    assert block["last_iteration_comparison"]["rms"] > 0


def test_final_matches_no_stored_iteration(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "BVO" / "BVO_beta40_nwarmup100000.h5"
    _write_run(path, final_source=None)

    row = classify_hdf5_run(path, root)

    assert row["run_status"] == "final_matches_no_stored_iteration"
    block = row["G_iw_storage"]["blocks"][0]
    assert block["exact_match_iterations"] == []
    assert block["closest_stored_iteration"] == 15
    assert block["closest_stored_iteration_metrics"]["mean_abs"] > 0
    assert block["relative_rms_floor"] == 1e-12


def test_beta_variation_is_parameter_candidate_and_reports_written_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    path = root / "SNO" / "beta-variation" / "SNO_beta20_nwarmup150000.h5"
    _write_run(path, final_source=15)

    row = classify_hdf5_run(path, root)
    summary = classify_hdf5_runs(root, tmp_path / "reports", low_frequency_count=3)

    assert row["parameter_variation_candidate"] is True
    assert row["run_status"] == "final_matches_last_iteration"
    assert summary["file_count"] == 1
    assert summary["diagnostic_scope"] == "storage_consistency_not_dmft_convergence"
    assert (tmp_path / "reports" / "run_diagnostics.json").is_file()
    assert (tmp_path / "reports" / "run_diagnostics.csv").is_file()


def test_classifies_string_complex_marker(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "SVO" / "SVO_beta40.h5"
    _write_run(path, final_source=15)

    row = classify_hdf5_run(path, root)

    assert row["run_status"] == "final_matches_last_iteration"


def test_storage_cli_matches_legacy_script(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_run(root / "SVO" / "run.h5", final_source=15)
    script = Path(__file__).resolve().parents[1] / "scripts/07_classify_hdf5_runs.py"
    outputs = [tmp_path / "cli", tmp_path / "legacy"]
    for command, output in zip(
        ([sys.executable, "-m", "dmft_data", "storage"], [sys.executable, str(script)]),
        outputs,
    ):
        result = subprocess.run(
            [*command, "--source", str(root), "--output", str(output),
             "--low-frequency-count", "3"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
    for name in ("run_diagnostics.json", "run_diagnostics_summary.json"):
        assert json.loads((outputs[0] / name).read_text()) == json.loads(
            (outputs[1] / name).read_text()
        )


def test_storage_source_must_exist(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="source directory"):
        classify_hdf5_runs(tmp_path / "missing", tmp_path / "out")
    assert not (tmp_path / "out").exists()

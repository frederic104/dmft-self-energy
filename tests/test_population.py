from pathlib import Path

import h5py
import numpy as np

from dmft_data.population import summarize_compatibility_canonical


def test_summarizes_record_axes_without_canonicalizing_channels(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.h5"
    with h5py.File(canonical, "w") as handle:
        states = handle.create_group("physical_states")
        states.create_group("state-A")
        runs = handle.create_group("runs")
        for index in range(2):
            group = runs.create_group(f"record_{index:06d}")
            group.attrs["physical_state_id"] = "state-A"
            group.attrs["run_id"] = "run-A"
            group.attrs["iteration"] = str(index + 1)
            group.attrs["role"] = "impurity_self_energy"
            group.attrs["source_path"] = "/dmft/Sigma"
            group.attrs["source_sha256"] = "a" * 64
            group.attrs["warmup"] = False
            group.attrs["spin_labels"] = ["up", "down"]
            group.attrs["orbital_labels"] = ["dxz", "dyz"]
            group.attrs["metadata_temperature"] = "300"
            group.attrs["frequency_axis"] = 0
            group.create_dataset("values", data=np.ones((3, 2), dtype=np.complex128))
            mesh = group.create_dataset("matsubara_mesh", data=np.array([1j, 3j, 5j]))
            mesh.attrs["source_path"] = "/iw"

    output = tmp_path / "population.json"
    report = summarize_compatibility_canonical(canonical, output)

    assert report["status"] == "passed"
    assert report["record_count"] == 2
    assert report["run_count"] == 1
    assert report["role_counts"] == {"impurity_self_energy": 2}
    assert report["iteration_counts"] == {"1": 1, "2": 1}
    assert report["spin_label_counts"] == {"up": 2, "down": 2}
    assert report["orbital_label_counts"] == {"dxz": 2, "dyz": 2}
    assert report["mesh_shape_counts"] == {"(3,)": 2}
    assert report["mesh_positive_imag_count_distribution"] == {"3": 2}
    assert report["duplicate_exact_identity_count"] == 0
    assert output.is_file()


def test_missing_canonical_is_reported_without_inventing_population(tmp_path: Path) -> None:
    output = tmp_path / "population.json"
    report = summarize_compatibility_canonical(tmp_path / "absent.h5", output)

    assert report["status"] == "missing_input"
    assert report["record_count"] == 0
    assert "not present" in report["limitations"][0]
    assert output.is_file()

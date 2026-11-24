"""Disentangle stored real shifts and imaginary changes without convergence claims."""

import h5py
import numpy as np
import pytest
from test_learning import setup_experiment

from dmft_data.iteration_analysis import metadata_candidates
from dmft_data.physics import inspect_raw


def test_exact_metadata_names_and_bounded_values(tmp_path):
    setup_experiment(tmp_path)
    path = tmp_path / "raw/BVO/run0.h5"
    with h5py.File(path, "r+") as handle:
        handle.attrs["iterations"] = 15
        handle.create_dataset("parameters/chemical_potential", data=0.25)
        handle.create_dataset("parameters/U", data=np.arange(100))
        handle.create_dataset("parameters/user", data="not a U parameter")
        rows = metadata_candidates(handle)
    assert any(r.get("attribute") == "iterations" and r["value"] == 15 for r in rows)
    assert any(
        r["source_path"] == "/parameters/chemical_potential" and r["value"] == 0.25
        for r in rows
    )
    large = next(r for r in rows if r["source_path"] == "/parameters/U")
    assert large["value_status"] == "not_read_large_metadata_candidate"
    assert not any(r["source_path"].endswith("/user") for r in rows)


def test_irregular_iteration_numbers_real_shift_and_imaginary_shape(tmp_path):
    setup_experiment(tmp_path)
    path = tmp_path / "raw/BVO/run0.h5"
    with h5py.File(path, "r+") as handle:
        handle.copy("dmft_output/G0-1", "dmft_output/G_iw")
        handle.copy("dmft_output/Sigma-5", "dmft_output/Sigma-8")
        for block in ("down_0", "up_0"):
            data = handle[f"dmft_output/Sigma-8/{block}/data"]
            values = data[()]
            values[..., 0] += 3
            data[...] = values
        handle.copy("dmft_output/Sigma-8", "dmft_output/Sigma-9")
        for block in ("down_0", "up_0"):
            data = handle[f"dmft_output/Sigma-9/{block}/data"]
            values = data[()]
            values[..., 1] += np.arange(16)[:, None, None]
            data[...] = values
    report = inspect_raw(path, 4, tmp_path, "BVO")
    rows = {
        r["iteration"]: r
        for r in report["iteration_comparisons"]
        if r["quantity"] == "Sigma"
    }
    assert rows[8]["previous_stored_iteration"] == 5
    delta = rows[8]["change_from_previous"]
    assert delta["real_offset_rms"] == pytest.approx(3)
    assert delta["real_shape_rmse"] < 1e-12
    assert delta["imaginary_rmse"] == 0
    assert rows[9]["change_from_previous"]["imaginary_rmse"] > 0
    assert rows[9]["change_from_previous"]["real_rmse"] == 0
    assert report["convergence_status"] == "not_assessed"
    assert (tmp_path / "BVO_step_changes.csv").is_file()
    assert (tmp_path / "BVO_step_changes.png").stat().st_size > 1000
    assert (tmp_path / "BVO_step_changes.pdf").is_file()
    assert (
        "previous STORED iteration"
        in (tmp_path / "BVO_step_changes.caption.md").read_text()
    )
    assert (tmp_path / "BVO_iterations.pdf").is_file()

"""Definition, unit, oracle-error and same-stage Dyson diagnostic regressions."""

import h5py
import numpy as np
import pytest
import yaml
from test_learning import setup_experiment

from dmft_data.hdf import read_complex_dataset
from dmft_data.learning.data import prepare
from dmft_data.learning.experiment import run_experiment
from dmft_data.physics import decompose, diagnose, dyson_check, inspect_raw
from dmft_data.provenance import sha256_file


def test_signed_offset_is_not_predictive_accuracy():
    reference = np.ones((2, 1, 1, 4), dtype=complex) * -1j
    result = decompose(reference + 2 + 3j, reference)
    assert result["real_offset_rms"] == 2
    assert np.all(np.asarray(result["signed_real_offset"]) == 2)
    assert result["complex_rmse"] == pytest.approx(np.sqrt(13))
    assert result["oracle_centered_complex_rmse"] == 3


def test_reports_and_plots_without_assuming_energy_units(tmp_path):
    config = setup_experiment(tmp_path)
    config["models"] = [{"name": "nearest"}]
    prepare(config, tmp_path)
    run_experiment(config, tmp_path)
    report = diagnose(config, tmp_path)
    output = tmp_path / "experiment/physics_diagnostics"
    assert report["native_energy_unit"] is None
    assert report["nearest_groups"][0]["neighbors"]
    assert report["convergence_status"] == "not_assessed"
    assert (output / "block_metrics.csv").is_file()
    assert (output / "validation_plan.json").is_file()
    assert (output / "seed_3_group_0.pdf").is_file()
    assert (output / "seed_3_group_0_blocks.pdf").is_file()
    tasks = {t["check"]: t for t in report["validation_tasks"]}
    assert tasks["DMFT_self_consistency"]["status"] == "not_assessed"
    assert tasks["quantity_basis_energy_conventions"]["status"] == "unresolved"
    assert (output / "seed_3_group_0.png").stat().st_size > 1000
    assert "not predictive performance" in (output / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        diagnose(config, tmp_path)
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text(
        yaml.safe_dump(
            {
                "definitions": {
                    "energy_unit": {
                        "value": "eV",
                        "status": "reviewed",
                        "evidence": "synthetic units",
                    }
                }
            }
        )
    )
    confirmed = diagnose(config, tmp_path, definitions, tmp_path / "confirmed")
    assert confirmed["plot_unit"] == "meV"
    assert confirmed["metric_unit"] == "native"
    assert confirmed["seeds"] == report["seeds"]
    definitions.write_text(
        yaml.safe_dump(
            {
                "definitions": {
                    "energy_unit": {
                        "value": "eV",
                        "status": "reviewed",
                        "evidence": None,
                    }
                }
            }
        )
    )
    unverified = diagnose(config, tmp_path, definitions, tmp_path / "unverified")
    assert unverified["native_energy_unit"] is None


def test_dyson_requires_scoped_semantics_and_stage(tmp_path):
    setup_experiment(tmp_path)
    path = tmp_path / "raw/BVO/run0.h5"
    with h5py.File(path, "r+") as handle:
        handle.copy("dmft_output/G0-1", "dmft_output/G-5")
        for block in ("down_0", "up_0"):
            w = read_complex_dataset(handle[f"dmft_output/G0-1/{block}/data"])
            sigma = read_complex_dataset(handle[f"dmft_output/Sigma-5/{block}/data"])
            g = 1 / (1 / w - sigma)
            handle[f"dmft_output/G-5/{block}/data"][...] = np.stack(
                [g.real, g.imag], axis=-1
            )
        handle.copy("dmft_output/G-5", "dmft_output/G_iw")
    spec = {
        "source_sha256": sha256_file(path),
        "weiss_path": "/dmft_output/G0-1",
        "green_path": "/dmft_output/G-5",
        "sigma_path": "/dmft_output/Sigma-5",
        "weiss_semantics": "dmft_weiss",
        "green_semantics": "impurity_green",
        "sigma_semantics": "impurity_self_energy",
        "sigma_convention": "solver_without_double_counting_subtraction",
        "same_stage_evidence": "synthetic construction only",
    }
    assert dyson_check(path, spec, 4)["rms_residual"] < 1e-12
    assert (
        dyson_check(path, {**spec, "weiss_semantics": "dft_derived_g0"}, 4)["status"]
        == "not_assessed"
    )
    assert (
        dyson_check(path, {**spec, "source_sha256": "stale"}, 4)["status"]
        == "not_assessed"
    )
    report = inspect_raw(path, 4, tmp_path, "BVO")
    sigma = [r for r in report["iteration_comparisons"] if r["quantity"] == "Sigma"]
    assert sigma[0]["exact_final"] is True
    assert sigma[1]["exact_final"] is False
    assert len(report["spin_comparisons"]) == 2

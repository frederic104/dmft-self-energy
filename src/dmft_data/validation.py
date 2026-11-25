"""Numerical and structural validation for extracted DMFT data."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .convergence import assess_convergence
from .extraction import ExtractedArray, HDF5Extraction
from .hdf import difference_metrics


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source_path": self.source_path,
        }


@dataclass
class NumericalValidationReport:
    source: str
    sha256: str
    status: str
    issues: list[ValidationIssue] = field(default_factory=list)
    arrays: list[dict[str, Any]] = field(default_factory=list)
    equivalent_dataset_comparisons: list[dict[str, Any]] = field(default_factory=list)
    iteration_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    physical_convergence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "arrays": self.arrays,
            "equivalent_dataset_comparisons": self.equivalent_dataset_comparisons,
            "iteration_diagnostics": self.iteration_diagnostics,
            "physical_convergence": self.physical_convergence,
        }


def _array_validation(array: ExtractedArray) -> tuple[list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []
    values = np.asarray(array.values)
    if values.dtype.kind != "c":
        issues.append(
            ValidationIssue(
                "error", "dtype_not_complex", "Extracted array is not complex-valued",
                array.source_path,
            )
        )
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        issues.append(
            ValidationIssue(
                "error", "non_finite_values", "Array contains non-finite values",
                array.source_path,
            )
        )
    if values.ndim < 1:
        issues.append(
            ValidationIssue("error", "invalid_shape", "Array has no dimensions",
                            array.source_path)
        )

    mesh = array.frequency_mesh
    mesh_report: dict[str, Any] = {}
    if mesh is None:
        issues.append(
            ValidationIssue(
                "error", "missing_frequency_mesh", "No verified frequency mesh",
                array.source_path,
            )
        )
    else:
        mesh_values = np.asarray(mesh.values)
        mesh_report = {
            "source_path": mesh.source_path,
            "shape": list(mesh_values.shape),
            "dtype": str(mesh_values.dtype),
            "finite": bool(np.all(np.isfinite(mesh_values))),
        }
        if mesh_values.ndim != 1:
            issues.append(
                ValidationIssue("error", "invalid_frequency_mesh",
                                "Frequency mesh is not one-dimensional",
                                mesh.source_path)
            )
        if not np.all(np.isfinite(mesh_values)):
            issues.append(
                ValidationIssue("error", "non_finite_frequency_mesh",
                                "Frequency mesh contains non-finite values",
                                mesh.source_path)
            )
        if array.frequency_axis is None:
            issues.append(
                ValidationIssue("error", "missing_frequency_axis",
                                "Frequency axis is unresolved", array.source_path)
            )
        elif values.shape[array.frequency_axis] != mesh_values.size:
            issues.append(
                ValidationIssue(
                    "error", "frequency_mesh_shape_mismatch",
                    f"Array shape {values.shape} conflicts with mesh length "
                    f"{mesh_values.size} on axis {array.frequency_axis}",
                    array.source_path,
                )
            )

    return issues, {
        "source_path": array.source_path,
        "role": array.role,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "stored_dtype": array.stored_dtype,
        "iteration": array.iteration,
        "spin_labels": array.spin_labels,
        "orbital_labels": array.orbital_labels,
        "frequency_axis": array.frequency_axis,
        "finite": bool(np.all(np.isfinite(values))),
        "frequency_mesh": mesh_report,
    }


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    return {
        **difference_metrics(left, right),
        "equivalent_at_1e-12": bool(np.allclose(left, right, rtol=1e-12, atol=1e-12)),
    }


def validate_extraction(
    extraction: HDF5Extraction,
    validation_config: dict[str, Any] | None = None,
) -> NumericalValidationReport:
    """Perform numerical validation and diagnostics only."""
    config = validation_config or {}
    issues: list[ValidationIssue] = []
    array_reports: list[dict[str, Any]] = []
    for array in extraction.arrays:
        array_issues, array_report = _array_validation(array)
        issues.extend(array_issues)
        array_reports.append(array_report)

    if not extraction.arrays:
        issues.append(
            ValidationIssue(
                "error", "missing_datasets",
                "No supported Green function or self-energy datasets were extracted",
                "/",
            )
        )

    self_energies = [array for array in extraction.arrays if "self_energy" in array.role]
    equivalent: list[dict[str, Any]] = []
    for index, left in enumerate(self_energies):
        for right in self_energies[index + 1:]:
            if left.values.shape != right.values.shape:
                continue
            if left.frequency_axis != right.frequency_axis:
                continue
            metric = _difference(left.values, right.values)
            equivalent.append({
                "left_source_path": left.source_path,
                "right_source_path": right.source_path,
                "left_iteration": left.iteration,
                "right_iteration": right.iteration,
                **metric,
            })

    iteration_diagnostics: list[dict[str, Any]] = []
    for role in sorted({array.role for array in self_energies}):
        by_iteration: dict[str, list[ExtractedArray]] = {}
        for array in self_energies:
            if array.role == role and array.iteration is not None:
                by_iteration.setdefault(array.iteration, []).append(array)
        ordered = sorted(by_iteration)
        for left_id, right_id in itertools.pairwise(ordered):
            for left in by_iteration[left_id]:
                for right in by_iteration[right_id]:
                    if left.values.shape != right.values.shape:
                        continue
                    if left.spin_labels != right.spin_labels:
                        continue
                    if left.orbital_labels != right.orbital_labels:
                        continue
                    metric = _difference(left.values, right.values)
                    iteration_diagnostics.append({
                        "role": role,
                        "from_iteration": left_id,
                        "to_iteration": right_id,
                        "left_source_path": left.source_path,
                        "right_source_path": right.source_path,
                        **metric,
                    })

    error_count = sum(issue.severity == "error" for issue in issues)
    return NumericalValidationReport(
        source=extraction.source_path,
        sha256=extraction.sha256,
        status="failed" if error_count else "passed",
        issues=issues,
        arrays=array_reports,
        equivalent_dataset_comparisons=equivalent,
        iteration_diagnostics=iteration_diagnostics,
        physical_convergence=assess_convergence(config),
    )


# Backward-friendly public alias.
validate_extracted_data = validate_extraction


def run_validation(
    config_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate extracted source files and write machine-readable reports."""
    config_path = Path(config_path)
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.resolve().parent.parent
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")

    dataset_path = root / "configs" / "dataset.yaml"
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)
    if not isinstance(dataset_config, dict):
        raise TypeError(f"Configuration must be a mapping: {dataset_path}")

    raw_dir = root / dataset_config["paths"]["raw_dir"]
    report_dir = root / dataset_config["paths"]["report_dir"]
    validation_dir = report_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in dataset_config["dataset"].get(
            "file_extensions", [".h5", ".hdf5"]
        )
    }
    recursive = dataset_config["dataset"].get("recursive", True)
    sources = []
    if raw_dir.is_dir():
        candidates = raw_dir.rglob("*") if recursive else raw_dir.iterdir()
        sources = sorted(
            (
                path
                for path in candidates
                if path.is_file() and path.suffix.lower() in extensions
            ),
            key=lambda path: path.as_posix(),
        )

    aggregate: dict[str, Any] = {
        "status": "passed" if raw_dir.is_dir() else "failed",
        "raw_dir": str(raw_dir.resolve()),
        "validation_config": config,
        "files": [],
        "failures": [],
    }
    if not raw_dir.is_dir():
        aggregate["failures"].append({
            "relative_path": str(dataset_config["paths"]["raw_dir"]),
            "error_type": "FileNotFoundError",
            "error": f"Configured raw data directory does not exist: {raw_dir}",
        })

    from .extraction import read_hdf5_file

    for source in sources:
        relative = source.relative_to(raw_dir).as_posix()
        try:
            extraction = read_hdf5_file(source, raw_dir)
            report = validate_extraction(extraction, config)
            report_data = report.to_dict()
            individual = validation_dir / Path(relative).with_suffix(".json")
            individual.parent.mkdir(parents=True, exist_ok=True)
            individual.write_text(
                json.dumps(report_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            aggregate["files"].append({
                "relative_source_path": relative,
                "source_sha256": extraction.sha256,
                "report_path": str(individual.resolve()),
                "status": report.status,
                "issue_count": len(report.issues),
            })
            if report.status == "failed":
                aggregate["status"] = "failed"
        except Exception as error:  # noqa: BLE001
            aggregate["status"] = "failed"
            aggregate["failures"].append({
                "relative_path": relative,
                "source_path": str(source.resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
            })

    aggregate_path = report_dir / "hdf5_validation.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate

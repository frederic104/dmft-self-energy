"""Collect inventory and numerical reports from local HDF5 files.

This script reads source HDF5 files without modifying or copying their payload.
It creates machine-readable reports that can be transferred back to this
repository for schema review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dmft_data.extraction import read_hdf5_file, software_versions
from dmft_data.inventory import inspect_hdf5_file
from dmft_data.triqs_inspection import inspect_triqs_file
from dmft_data.validation import validate_extraction


LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without changing the source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_statistics(values: np.ndarray) -> dict[str, Any]:
    real = np.asarray(values.real)
    imaginary = np.asarray(values.imag)
    return {
        "real_min": float(np.min(real)),
        "real_max": float(np.max(real)),
        "imaginary_min": float(np.min(imaginary)),
        "imaginary_max": float(np.max(imaginary)),
    }


def inspect_one_file(
    source: Path,
    source_root: Path,
    per_file_dir: Path,
) -> dict[str, Any]:
    """Create inventory, extraction, and validation reports for one source."""
    relative = source.relative_to(source_root).as_posix()
    result: dict[str, Any] = {
        "relative_source_path": relative,
        "sha256": sha256_file(source),
        "status": "success",
        "inventory": {},
        "extraction": {},
        "validation": {},
        "triqs": {},
        "failures": [],
    }

    file_row, group_rows, dataset_rows, inventory_failures = inspect_hdf5_file(
        source, source_root
    )
    result["inventory"] = {
        "file": file_row,
        "groups": group_rows,
        "datasets": dataset_rows,
        "failures": inventory_failures,
    }
    result["failures"].extend(
        {"stage": "inventory", **failure} for failure in inventory_failures
    )

    try:
        extraction = read_hdf5_file(source, source_root)
        arrays: list[dict[str, Any]] = []
        for array in extraction.arrays:
            mesh = None
            if array.frequency_mesh is not None:
                mesh_values = np.asarray(array.frequency_mesh.values)
                mesh = {
                    "source_path": array.frequency_mesh.source_path,
                    "shape": list(mesh_values.shape),
                    "dtype": str(mesh_values.dtype),
                    "finite": bool(np.all(np.isfinite(mesh_values))),
                }
            arrays.append({
                "role": array.role,
                "source_path": array.source_path,
                "shape": list(array.values.shape),
                "stored_dtype": array.stored_dtype,
                "dtype": str(array.values.dtype),
                "finite": bool(np.all(np.isfinite(array.values))),
                "frequency_axis": array.frequency_axis,
                "iteration": array.iteration,
                "spin_labels": array.spin_labels,
                "orbital_labels": array.orbital_labels,
                "component_statistics": _component_statistics(array.values),
                "frequency_mesh": mesh,
                "attributes": array.attributes,
            })
        result["extraction"] = {
            "mesh_count": len(extraction.meshes),
            "array_count": len(extraction.arrays),
            "meshes": [
                {
                    "source_path": mesh.source_path,
                    "shape": list(np.asarray(mesh.values).shape),
                    "dtype": str(np.asarray(mesh.values).dtype),
                    "finite": bool(np.all(np.isfinite(mesh.values))),
                }
                for mesh in extraction.meshes
            ],
            "arrays": arrays,
        }

        validation = validate_extraction(extraction)
        result["validation"] = validation.to_dict()
        if validation.status == "failed":
            result["status"] = "failed"
    except Exception as error:  # noqa: BLE001
        result["status"] = "failed"
        result["failures"].append({
            "stage": "extraction_or_validation",
            "error_type": type(error).__name__,
            "error": str(error),
        })

    try:
        result["triqs"] = inspect_triqs_file(source, source_root)
    except Exception as error:  # noqa: BLE001
        result["status"] = "failed"
        result["failures"].append({
            "stage": "triqs",
            "error_type": type(error).__name__,
            "error": str(error),
        })

    per_file_dir.mkdir(parents=True, exist_ok=True)
    report_path = per_file_dir / Path(relative).with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["report_path"] = str(report_path)
    return result


def collect_laptop_results(
    source_root: Path,
    output_dir: Path,
    extensions: tuple[str, ...] = (".h5", ".hdf5"),
) -> tuple[dict[str, Any], Path]:
    """Collect a transfer bundle for one local HDF5 source directory."""
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = output_dir / "per_file"

    normalized_extensions = tuple(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    )
    if source_root.is_dir():
        sources = sorted(
            (
                path
                for path in source_root.rglob("*")
                if path.is_file() and path.suffix.lower() in normalized_extensions
            ),
            key=lambda path: path.as_posix(),
        )
    else:
        sources = []

    files: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for source in sources:
        try:
            report = inspect_one_file(source, source_root, per_file_dir)
        except Exception as error:  # noqa: BLE001
            relative = source.relative_to(source_root).as_posix()
            report = {
                "relative_source_path": relative,
                "sha256": sha256_file(source),
                "status": "failed",
                "failures": [{
                    "stage": "bundle",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }],
            }
            files.append(report)
        else:
            files.append(report)

        csv_rows.append({
            "relative_source_path": report["relative_source_path"],
            "sha256": report["sha256"],
            "status": report["status"],
            "inventory_datasets": len(report.get("inventory", {}).get("datasets", [])),
            "extracted_arrays": report.get("extraction", {}).get("array_count", 0),
            "failure_count": len(report.get("failures", [])),
        })

    aggregate = {
        "status": "passed" if source_root.is_dir() and not any(
            file["status"] == "failed" for file in files
        ) else "failed",
        "source_root": str(source_root),
        "source_count": len(sources),
        "software": software_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "files": files,
    }
    if not source_root.is_dir():
        aggregate["failures"] = [{
            "stage": "source",
            "error_type": "FileNotFoundError",
            "error": f"Source directory does not exist: {source_root}",
        }]
    else:
        aggregate["failures"] = []

    aggregate_path = output_dir / "laptop_collection.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "laptop_collection.csv"
    fieldnames = [
        "relative_source_path",
        "sha256",
        "status",
        "inventory_datasets",
        "extracted_arrays",
        "failure_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    zip_path = output_dir / "laptop_collection.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(output_dir).as_posix())

    LOGGER.info(
        "Collected %d HDF5 files; status=%s; bundle=%s",
        len(sources),
        aggregate["status"],
        zip_path,
    )
    return aggregate, zip_path


def main() -> None:
    """Run the local collection bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local directory containing the original HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("laptop-results"),
        help="Directory for reports and the ZIP bundle.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    collect_laptop_results(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

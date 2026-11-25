"""Non-destructive inventory of local DFT+DMFT HDF5 files.

This module records HDF5 structure and provenance only. It never reads dataset
payload arrays and it does not infer physical meanings for ambiguous names.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .provenance import sha256_file

CSV_FIELDS = {
    "files": (
        "relative_path",
        "sha256",
        "status",
        "file_attributes",
        "error",
    ),
    "groups": ("relative_path", "group_path", "attributes"),
    "datasets": (
        "relative_path",
        "dataset_path",
        "shape",
        "dtype",
        "attributes",
        "candidate_categories",
    ),
    "failures": ("relative_path", "object_path", "error_type", "error"),
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a dataset YAML configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return config


def discover_hdf5_files(
    raw_dir: str | Path,
    extensions: Iterable[str],
    recursive: bool = True,
) -> list[Path]:
    """Discover source HDF5 files under a directory."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Configured raw data directory does not exist: {root}")
    normalized = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }

    if recursive:
        candidates = (path for path in root.rglob("*") if path.is_file())
    else:
        candidates = (path for path in root.iterdir() if path.is_file())
    return sorted(
        (path for path in candidates if path.suffix.lower() in normalized),
        key=lambda path: path.as_posix(),
    )


def json_value(value: Any) -> Any:
    """Convert an HDF5 attribute value to a JSON-compatible value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def attributes_json(mapping: Any) -> str:
    """Serialize an HDF5 attribute mapping reproducibly."""
    attributes = {key: json_value(value) for key, value in mapping.items()}
    return json.dumps(attributes, sort_keys=True, ensure_ascii=True)


def candidate_categories(dataset_path: str) -> str:
    """Identify name-based candidates without assigning physical semantics."""
    name = dataset_path.lower()
    categories: list[str] = []

    green_terms = (
        "green",
        "/g0",
        "/gimp",
        "/g_lat",
        "/g0imp",
        "/weiss",
        "/weiss_field",
        "/hybrid",
    )
    sigma_terms = ("sigma", "self_energy", "selfenergy")
    frequency_terms = (
        "omega",
        "freq",
        "frequency",
        "matsubara",
        "mesh",
    )

    if any(term in name for term in green_terms) or name.rstrip("/").endswith("/g"):
        categories.append("candidate_green_function")
    if any(term in name for term in sigma_terms):
        categories.append("candidate_self_energy")
    if any(term in name for term in frequency_terms):
        categories.append("candidate_frequency_mesh")
    return ";".join(categories)


def _append_failure(
    failures: list[dict[str, str]],
    relative_path: str,
    object_path: str,
    error: Exception,
) -> None:
    """Append a machine-readable failure row."""
    failures.append(
        {
            "relative_path": relative_path,
            "object_path": object_path,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def inspect_hdf5_file(
    file_path: str | Path,
    source_root: str | Path,
) -> tuple[
    dict[str, str], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]:
    """Inspect one HDF5 file without reading dataset payloads."""
    path = Path(file_path)
    root = Path(source_root)
    relative_path = path.relative_to(root).as_posix()
    checksum = sha256_file(path)
    file_rows: dict[str, str] = {
        "relative_path": relative_path,
        "sha256": checksum,
        "status": "success",
        "file_attributes": "{}",
        "error": "",
    }
    group_rows: list[dict[str, str]] = []
    dataset_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    try:
        with h5py.File(path, "r") as handle:
            try:
                file_rows["file_attributes"] = attributes_json(handle.attrs)
            except Exception as error:  # noqa: BLE001
                _append_failure(failures, relative_path, "/", error)
            try:
                group_rows.append(
                    {
                        "relative_path": relative_path,
                        "group_path": "/",
                        "attributes": attributes_json(handle.attrs),
                    }
                )
            except Exception as error:  # noqa: BLE001
                _append_failure(failures, relative_path, "/", error)

            def visit(name: str, object_: h5py.Group | h5py.Dataset) -> None:
                object_path = f"/{name}" if name else "/"
                try:
                    if isinstance(object_, h5py.Group):
                        group_rows.append(
                            {
                                "relative_path": relative_path,
                                "group_path": object_path,
                                "attributes": attributes_json(object_.attrs),
                            }
                        )
                    elif isinstance(object_, h5py.Dataset):
                        dataset_rows.append(
                            {
                                "relative_path": relative_path,
                                "dataset_path": object_path,
                                "shape": json.dumps(list(object_.shape)),
                                "dtype": str(object_.dtype),
                                "attributes": attributes_json(object_.attrs),
                                "candidate_categories": candidate_categories(
                                    object_path
                                ),
                            }
                        )
                except Exception as error:  # noqa: BLE001
                    _append_failure(failures, relative_path, object_path, error)

            handle.visititems(visit)
    except Exception as error:  # noqa: BLE001
        file_rows["status"] = "failed"
        file_rows["error"] = str(error)
        _append_failure(failures, relative_path, "/", error)

    return file_rows, group_rows, dataset_rows, failures


def write_csv(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]
) -> None:
    """Write CSV data atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_markdown(
    path: Path,
    file_rows: list[dict[str, str]],
    group_rows: list[dict[str, str]],
    dataset_rows: list[dict[str, str]],
    failure_rows: list[dict[str, str]],
) -> None:
    """Write a concise Markdown inventory summary."""
    success_count = sum(row["status"] == "success" for row in file_rows)
    failed_count = sum(row["status"] == "failed" for row in file_rows)
    categories = sorted(
        {
            category
            for row in dataset_rows
            for category in row["candidate_categories"].split(";")
            if category
        }
    )

    lines = [
        "# DFT+DMFT HDF5 Inventory",
        "",
        "This inventory records HDF5 structure and provenance only.",
        "It does not extract scientific arrays or infer physical meanings.",
        "",
        "## Summary",
        "",
        f"- Files discovered: {len(file_rows)}",
        f"- Successfully opened: {success_count}",
        f"- Failed: {failed_count}",
        f"- Groups: {len(group_rows)}",
        f"- Datasets: {len(dataset_rows)}",
        f"- Failures: {len(failure_rows)}",
        "",
        "## Name-Based Dataset Candidates",
        "",
    ]
    if categories:
        lines.extend(f"- `{category}`" for category in categories)
    else:
        lines.append("- No name-based candidates identified.")

    lines.extend(
        [
            "",
            "## File Status",
            "",
            "| Relative source path | Status | SHA-256 | Error |",
            "|---|---|---|---|",
        ]
    )
    for row in file_rows:
        status = row["status"]
        error = row["error"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{row['relative_path']}` | {status} | `{row['sha256']}` | {error} |"
        )

    lines.extend(
        [
            "",
            "## Unresolved Schema Questions",
            "",
        ]
    )
    lines.append(
        "- Dataset names are candidates only; physical meanings, channels, and iterations "
        "must be established from schema-specific metadata."
    )
    lines.append(
        "- Convergence status is unknown unless explicitly represented in file or "
        "dataset attributes and independently validated."
    )
    lines.append(
        "- Complex representations, frequency conventions, and spin/orbital labels "
        "cannot be inferred by inventory inspection alone."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_inventory(
    config_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Run HDF5 inventory and write machine-readable and Markdown reports."""
    config_path = Path(config_path)
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.resolve().parent.parent
    )
    config = load_config(config_path)

    paths_config = config["paths"]
    raw_dir = root / paths_config["raw_dir"]
    metadata_dir = root / paths_config["metadata_dir"]
    report_dir = root / paths_config["report_dir"]
    dataset_config = config["dataset"]

    file_rows: list[dict[str, str]] = []
    group_rows: list[dict[str, str]] = []
    dataset_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []

    try:
        files = discover_hdf5_files(
            raw_dir,
            dataset_config.get("file_extensions", (".h5", ".hdf5")),
            dataset_config.get("recursive", True),
        )
    except FileNotFoundError as error:
        files = []
        relative_path = str(Path(paths_config["raw_dir"]).as_posix())
        file_rows.append(
            {
                "relative_path": relative_path,
                "sha256": "",
                "status": "failed",
                "file_attributes": "{}",
                "error": str(error),
            }
        )
        failure_rows.append(
            {
                "relative_path": relative_path,
                "object_path": "/",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    for file_path in files:
        file_row, groups, datasets, failures = inspect_hdf5_file(file_path, raw_dir)
        file_rows.append(file_row)
        group_rows.extend(groups)
        dataset_rows.extend(datasets)
        failure_rows.extend(failures)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(metadata_dir / "hdf5_files.csv", CSV_FIELDS["files"], file_rows)
    write_csv(metadata_dir / "hdf5_groups.csv", CSV_FIELDS["groups"], group_rows)
    write_csv(metadata_dir / "hdf5_datasets.csv", CSV_FIELDS["datasets"], dataset_rows)
    write_csv(metadata_dir / "hdf5_failures.csv", CSV_FIELDS["failures"], failure_rows)

    inventory_metadata = {
        "raw_dir": str(raw_dir.resolve()),
        "relative_source_root": str(Path(paths_config["raw_dir"]).as_posix()),
        "h5py_version": h5py.__version__,
        "numpy_version": np.__version__,
        "discovered_file_count": len(file_rows),
        "file_extensions": dataset_config.get("file_extensions", [".h5", ".hdf5"]),
        "recursive": dataset_config.get("recursive", True),
    }
    metadata_path = metadata_dir / "hdf5_inventory_metadata.json"
    metadata_path.write_text(
        json.dumps(inventory_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_markdown(
        report_dir / "hdf5_inventory.md",
        file_rows,
        group_rows,
        dataset_rows,
        failure_rows,
    )
    return {
        "files": file_rows,
        "groups": group_rows,
        "datasets": dataset_rows,
        "failures": failure_rows,
    }

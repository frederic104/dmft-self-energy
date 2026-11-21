"""Run-level HDF5 diagnostics for data validation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .extraction import read_hdf5_file


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _iteration_numbers(dmft: h5py.Group, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    numbers = []
    for name in dmft.keys():
        match = pattern.match(name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def _mesh_metadata(dmft: h5py.Group) -> dict[str, Any]:
    for name, obj in dmft.items():
        if not isinstance(obj, h5py.Group) or "-" not in name:
            continue
        candidates = list(obj.get("down_0/mesh", {}).keys()) if "down_0" in obj else []
        del candidates
        mesh = obj.get("down_0/mesh")
        if isinstance(mesh, h5py.Group):
            metadata: dict[str, Any] = {}
            for key in ("beta", "positive_freq_only", "size"):
                dataset = mesh.get(f"domain/{key}") if key == "beta" else mesh.get(key)
                if isinstance(dataset, h5py.Dataset) and dataset.shape == ():
                    value = dataset[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    metadata[key] = value.item() if isinstance(value, np.generic) else value
            statistic = mesh.get("domain/statistic")
            if isinstance(statistic, h5py.Dataset) and statistic.shape == ():
                try:
                    value = statistic[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    metadata["statistic"] = value
                except (TypeError, ValueError):
                    metadata["statistic"] = "unreadable"
            if metadata:
                return metadata
    return {}


def _complex_payload(dataset: h5py.Dataset) -> np.ndarray | None:
    raw = np.asarray(dataset[()])
    if (
        dataset.attrs.get("__complex__", 0) == 1
        or dataset.attrs.get("complex", 0) == 1
    ) and raw.ndim >= 1:
        if raw.shape[-1] == 2:
            return raw[..., 0] + 1j * raw[..., 1]
    if raw.ndim >= 1 and raw.shape[-1] == 2 and raw.dtype.kind in "iuf":
        return raw[..., 0] + 1j * raw[..., 1]
    if dataset.dtype.kind == "c":
        return raw
    return None


def _final_equivalence(
    dmft: h5py.Group,
    quantity: str,
    iterations: list[int],
) -> dict[str, Any]:
    final = dmft.get(f"{quantity}_iw")
    if not iterations or not isinstance(final, h5py.Group):
        return {
            "available": False,
            "last_iteration": iterations[-1] if iterations else None,
            "blocks": [],
            "reason": (
                "missing_iteration_groups"
                if not iterations
                else "missing_final_quantity"
            ),
        }

    last = iterations[-1]
    block_comparisons = []
    missing_blocks = []
    shape_mismatches = []
    unsupported_payloads = []
    for block_name, block in final.items():
        if not isinstance(block, h5py.Group):
            continue
        final_dataset = block.get("data")
        iteration_dataset = dmft.get(f"{quantity}-{last}/{block_name}/data")
        if not isinstance(final_dataset, h5py.Dataset):
            missing_blocks.append(block_name)
            continue
        if not isinstance(iteration_dataset, h5py.Dataset):
            missing_blocks.append(block_name)
            continue

        final_values = _complex_payload(final_dataset)
        iteration_values = _complex_payload(iteration_dataset)
        if final_values is None or iteration_values is None:
            unsupported_payloads.append(block_name)
            continue
        if final_values.shape != iteration_values.shape:
            shape_mismatches.append({
                "block": block_name,
                "final_shape": list(final_values.shape),
                "iteration_shape": list(iteration_values.shape),
            })
            continue

        difference = np.abs(final_values - iteration_values)
        block_comparisons.append({
            "block": block_name,
            "exact": bool(np.array_equal(final_values, iteration_values)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "rms": float(np.sqrt(np.mean(difference ** 2))),
        })

    return {
        "available": bool(block_comparisons),
        "last_iteration": last,
        "blocks": block_comparisons,
        "reason": None if block_comparisons else "no_comparable_blocks",
        "missing_blocks": missing_blocks,
        "shape_mismatches": shape_mismatches,
        "unsupported_payloads": unsupported_payloads,
    }


def classify_hdf5_run(
    path: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Classify one HDF5 run without convergence or mesh reconstruction."""
    source = Path(path)
    root = Path(source_root)
    relative = source.relative_to(root).as_posix()
    extraction = read_hdf5_file(source, root)

    with h5py.File(source, "r") as handle:
        dmft = handle.get("dmft_output")
        if not isinstance(dmft, h5py.Group):
            dmft_metadata: dict[str, Any] = {}
            g_iterations = g0_iterations = sigma_iterations = []
            g_equivalence = _final_equivalence(dmft, "G", []) if False else {
                "available": False,
                "last_iteration": None,
                "blocks": [],
            }
            sigma_equivalence = {
                "available": False,
                "last_iteration": None,
                "blocks": [],
            }
            dmft_status = "missing_dmft_output"
        else:
            dmft_metadata = _mesh_metadata(dmft)
            g_iterations = _iteration_numbers(dmft, "G")
            g0_iterations = _iteration_numbers(dmft, "G0")
            sigma_iterations = _iteration_numbers(dmft, "Sigma")
            g_equivalence = _final_equivalence(dmft, "G", g_iterations)
            sigma_equivalence = _final_equivalence(dmft, "Sigma", sigma_iterations)
            dmft_status = "present"

    material = relative.split("/", 1)[0] if "/" in relative else "unknown"
    warmup_match = re.search(r"nwarmup(\d+)", source.stem)
    warmup = int(warmup_match.group(1)) if warmup_match else None
    beta_variation = "/beta-variation/" in f"/{relative}"

    g_mismatch = g_equivalence["available"] and any(
        not block["exact"] for block in g_equivalence["blocks"]
    )
    sigma_mismatch = sigma_equivalence["available"] and any(
        not block["exact"] for block in sigma_equivalence["blocks"]
    )

    if g_mismatch or sigma_mismatch:
        run_status = "final_iteration_mismatch"
    elif beta_variation:
        run_status = "beta_variation_auxiliary"
    elif not g_equivalence["available"] or not sigma_equivalence["available"]:
        run_status = "final_quantity_unavailable"
    else:
        run_status = "extracted_final_equivalent"

    return {
        "relative_source_path": relative,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "material": material,
        "nwarmup": warmup,
        "beta_variation": beta_variation,
        "mesh_metadata": dmft_metadata,
        "mesh_status": "verified_1d_mesh_present" if extraction.meshes else "unresolved",
        "mesh_count": len(extraction.meshes),
        "array_count": len(extraction.arrays),
        "dmft_status": dmft_status,
        "G_iterations": g_iterations,
        "G0_iterations": g0_iterations,
        "Sigma_iterations": sigma_iterations,
        "G_iw_last": g_equivalence,
        "Sigma_iw_last": sigma_equivalence,
        "convergence_status": "not_assessed",
        "run_status": run_status,
        "primary_target_candidate": run_status
        == "extracted_final_equivalent",
    }


def classify_hdf5_runs(
    source_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Classify all HDF5 runs and write machine-readable reports."""
    root = Path(source_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    extensions = {".h5", ".hdf5"}
    sources = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    rows = [classify_hdf5_run(path, root) for path in sources]

    per_file_dir = output / "per_file"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        report_path = per_file_dir / Path(row["relative_source_path"]).with_suffix(".json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    aggregate_path = output / "run_diagnostics.json"
    aggregate_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["run_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "source_root": str(root),
        "file_count": len(rows),
        "run_status_counts": status_counts,
        "files_with_verified_mesh": sum(row["mesh_count"] > 0 for row in rows),
        "convergence_status": "not_assessed",
    }
    summary_path = output / "run_diagnostics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = output / "run_diagnostics.csv"
    fields = [
        "relative_source_path",
        "sha256",
        "material",
        "nwarmup",
        "beta_variation",
        "mesh_status",
        "mesh_count",
        "array_count",
        "run_status",
        "convergence_status",
        "primary_target_candidate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    return summary

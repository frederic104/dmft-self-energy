"""Storage-consistency diagnostics for DMFT-style HDF5 files.

These checks compare final arrays with stored iteration arrays. They do not
inspect residuals or solver logs and are not a DMFT convergence criterion.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .extraction import UnsupportedSchemaError, read_hdf5_file
from .hdf import physical_blocks as _physical_blocks
from .hdf import read_complex_dataset
from .provenance import sha256_file

DEFAULT_RELATIVE_RMS_FLOOR = 1e-12
DEFAULT_LOW_FREQUENCY_COUNT = 10


def _iteration_numbers(dmft: h5py.Group, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    return sorted({int(m.group(1)) for n in dmft for m in [pattern.match(n)] if m})


def _mesh_metadata(dmft: h5py.Group) -> dict[str, Any]:
    for obj in dmft.values():
        if not isinstance(obj, h5py.Group):
            continue
        mesh = obj.get("down_0/mesh")
        if not isinstance(mesh, h5py.Group):
            continue
        result: dict[str, Any] = {}
        for key in ("beta", "positive_freq_only", "size"):
            value = mesh.get(f"domain/{key}" if key == "beta" else key)
            if isinstance(value, h5py.Dataset) and value.shape == ():
                raw = value[()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                result[key] = raw.item() if isinstance(raw, np.generic) else raw
        if result:
            return result
    return {}


def _complex_payload(dataset: h5py.Dataset) -> np.ndarray | None:
    try:
        return read_complex_dataset(dataset)
    except UnsupportedSchemaError:
        return None


def _positive_frequency_indices(
    block: h5py.Group, axis_length: int, count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select first positive entries using the TRIQS BlockGf axis-0 convention."""
    positive_only = False
    mesh = block.get("mesh")
    if isinstance(mesh, h5py.Group):
        flag = mesh.get("positive_freq_only")
        if isinstance(flag, h5py.Dataset) and flag.shape == ():
            positive_only = bool(flag[()])
    start = 0 if positive_only else axis_length // 2
    stop = min(start + count, axis_length)
    return np.arange(start, stop), {
        "frequency_axis": 0,
        "positive_freq_only": positive_only,
        "requested_positive_frequency_count": count,
        "selected_indices": [int(start), int(stop)],
        "selection_rule": "first_positive_entries_on_triqs_axis_0",
    }


def _metrics(
    final: np.ndarray, candidate: np.ndarray, low_indices: np.ndarray, floor: float
) -> dict[str, float | None]:
    difference = np.abs(final - candidate)
    rms = float(np.sqrt(np.mean(difference**2)))
    reference_rms = float(np.sqrt(np.mean(np.abs(final) ** 2)))
    result: dict[str, float | None] = {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": rms,
        "relative_rms": rms / max(reference_rms, floor),
    }
    if low_indices.size:
        low_difference = difference[low_indices]
        low_reference = np.abs(final[low_indices])
        low_rms = float(np.sqrt(np.mean(low_difference**2)))
        result["low_frequency_relative_rms"] = low_rms / max(
            float(np.sqrt(np.mean(low_reference**2))), floor
        )
    else:
        result["low_frequency_relative_rms"] = None
    return result


def _final_equivalence(
    dmft: h5py.Group, quantity: str, iterations: list[int], *,
    low_frequency_count: int, relative_rms_floor: float,
) -> dict[str, Any]:
    """Compare every final physical block to every available stored iteration."""
    final_group = dmft.get(f"{quantity}_iw")
    last = iterations[-1] if iterations else None
    if not isinstance(final_group, h5py.Group):
        return {
            "status": "final_quantity_unavailable",
            "numerical_last_iteration": last,
            "blocks": [],
            "reason": "missing_final_quantity",
        }
    final_blocks = _physical_blocks(final_group)
    if not final_blocks:
        return {
            "status": "final_quantity_unavailable",
            "numerical_last_iteration": last,
            "blocks": [],
            "reason": "no_final_physical_data_blocks",
        }
    if not iterations:
        return {"status": "storage_schema_unresolved", "numerical_last_iteration": None,
                "blocks": [], "reason": "missing_iteration_groups"}

    blocks: list[dict[str, Any]] = []
    unresolved = False
    for name, final_dataset in final_blocks.items():
        final = _complex_payload(final_dataset)
        if final is None or final.ndim < 1:
            blocks.append({"block": name, "reason": "unsupported_final_payload"})
            unresolved = True
            continue
        indices, selection = _positive_frequency_indices(
            final_group[name], final.shape[0], low_frequency_count
        )
        comparisons: list[dict[str, Any]] = []
        unavailable: list[int] = []
        for iteration in iterations:
            candidate_dataset = dmft.get(f"{quantity}-{iteration}/{name}/data")
            candidate = (
                _complex_payload(candidate_dataset)
                if isinstance(candidate_dataset, h5py.Dataset)
                else None
            )
            if candidate is None or candidate.shape != final.shape:
                unavailable.append(iteration)
                continue
            comparisons.append({
                "iteration": iteration,
                "exact": bool(np.array_equal(final, candidate)),
                **_metrics(final, candidate, indices, relative_rms_floor),
            })
        if not comparisons:
            blocks.append({"block": name, "reason": "no_comparable_stored_iteration"})
            unresolved = True
            continue
        exact = [item["iteration"] for item in comparisons if item["exact"]]
        closest = min(comparisons, key=lambda item: (item["rms"], item["iteration"]))
        blocks.append({
            "block": name,
            "numerical_last_iteration": last,
            "last_iteration_comparison": next(
                (x for x in comparisons if x["iteration"] == last), None
            ),
            "closest_stored_iteration": closest["iteration"],
            "closest_stored_iteration_metrics": closest,
            "exact_match_iterations": exact,
            "iteration_comparisons": comparisons,
            "unavailable_iteration_comparisons": unavailable,
            "low_frequency_selection": selection,
            "relative_rms_floor": relative_rms_floor,
        })
        unresolved |= bool(unavailable)
    comparable = [block for block in blocks if "iteration_comparisons" in block]
    if unresolved or len(comparable) != len(final_blocks):
        status = "storage_schema_unresolved"
    elif all(last in block["exact_match_iterations"] for block in comparable):
        status = "final_matches_last_iteration"
    elif all(block["exact_match_iterations"] for block in comparable):
        status = "final_matches_earlier_iteration"
    else:
        status = "final_matches_no_stored_iteration"
    return {
        "status": status,
        "numerical_last_iteration": last,
        "blocks": blocks,
        "relative_rms_floor": relative_rms_floor,
        "low_frequency_count": low_frequency_count,
    }


def _run_status(g_status: str, sigma_status: str) -> str:
    statuses = {g_status, sigma_status}
    for status in (
        "storage_schema_unresolved",
        "final_quantity_unavailable",
        "final_matches_no_stored_iteration",
        "final_matches_earlier_iteration",
    ):
        if status in statuses:
            return status
    return "final_matches_last_iteration"


def classify_hdf5_run(
    path: str | Path, source_root: str | Path, *,
    low_frequency_count: int = DEFAULT_LOW_FREQUENCY_COUNT,
    relative_rms_floor: float = DEFAULT_RELATIVE_RMS_FLOOR,
) -> dict[str, Any]:
    """Diagnose HDF5 storage semantics; do not assess physical convergence."""
    if low_frequency_count <= 0 or relative_rms_floor <= 0:
        raise ValueError("low_frequency_count and relative_rms_floor must be positive")
    source, root = Path(path), Path(source_root)
    relative = source.relative_to(root).as_posix()
    extraction = read_hdf5_file(source, root)
    with h5py.File(source, "r") as handle:
        dmft = handle.get("dmft_output")
        if not isinstance(dmft, h5py.Group):
            dmft_metadata: dict[str, Any] = {}
            g_iterations = g0_iterations = sigma_iterations = []
            g_result = sigma_result = {
                "status": "storage_schema_unresolved",
                "numerical_last_iteration": None,
                "blocks": [],
                "reason": "missing_dmft_output",
            }
            dmft_status = "missing_dmft_output"
        else:
            dmft_metadata = _mesh_metadata(dmft)
            g_iterations = _iteration_numbers(dmft, "G")
            g0_iterations = _iteration_numbers(dmft, "G0")
            sigma_iterations = _iteration_numbers(dmft, "Sigma")
            g_result = _final_equivalence(
                dmft, "G", g_iterations,
                low_frequency_count=low_frequency_count,
                relative_rms_floor=relative_rms_floor,
            )
            sigma_result = _final_equivalence(
                dmft, "Sigma", sigma_iterations,
                low_frequency_count=low_frequency_count,
                relative_rms_floor=relative_rms_floor,
            )
            dmft_status = "present"
    warmup = re.search(r"nwarmup(\d+)", source.stem)
    return {
        "relative_source_path": relative, "sha256": sha256_file(source),
        "material": relative.split("/", 1)[0] if "/" in relative else "unknown",
        "nwarmup": int(warmup.group(1)) if warmup else None,
        "parameter_variation_candidate": "/beta-variation/" in f"/{relative}",
        "mesh_metadata": dmft_metadata,
        "mesh_status": (
            "verified_1d_mesh_present" if extraction.meshes else "unresolved"
        ),
        "mesh_count": len(extraction.meshes), "array_count": len(extraction.arrays),
        "dmft_status": dmft_status, "G_iterations": g_iterations,
        "G0_iterations": g0_iterations, "Sigma_iterations": sigma_iterations,
        "G_iw_storage": g_result, "Sigma_iw_storage": sigma_result,
        "convergence_status": "not_assessed",
        "run_status": _run_status(g_result["status"], sigma_result["status"]),
        "primary_target_candidate": False,
    }


def classify_hdf5_runs(
    source_root: str | Path, output_dir: str | Path, *,
    low_frequency_count: int = DEFAULT_LOW_FREQUENCY_COUNT,
    relative_rms_floor: float = DEFAULT_RELATIVE_RMS_FLOOR,
) -> dict[str, Any]:
    """Write storage-semantic reports for HDF5 runs below ``source_root``."""
    root, output = Path(source_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"HDF5 source directory does not exist: {root}")
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
    )
    rows = [
        classify_hdf5_run(
            p, root, low_frequency_count=low_frequency_count,
            relative_rms_floor=relative_rms_floor,
        )
        for p in sources
    ]
    per_file = output / "per_file"
    per_file.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = per_file / Path(row["relative_source_path"]).with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output / "run_diagnostics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["run_status"]] = counts.get(row["run_status"], 0) + 1
    summary = {
        "source_root": "<LOCAL_SOURCE_ROOT>",
        "file_count": len(rows),
        "run_status_counts": counts,
        "files_with_verified_mesh": sum(row["mesh_count"] > 0 for row in rows),
        "convergence_status": "not_assessed",
        "diagnostic_scope": "storage_consistency_not_dmft_convergence",
        "low_frequency_count": low_frequency_count,
        "relative_rms_floor": relative_rms_floor,
    }
    (output / "run_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "relative_source_path", "sha256", "material", "nwarmup",
        "parameter_variation_candidate", "mesh_status", "mesh_count",
        "array_count", "run_status", "convergence_status",
        "primary_target_candidate",
    ]
    with (output / "run_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return summary

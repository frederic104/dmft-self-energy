"""Read-only inspection helpers for TRIQS-style DMFT HDF5 files.

This module inventories known quantity and iteration paths, reports iteration
differences, and never assigns a convergence status without explicit metadata
or log evidence.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .hdf import difference_metrics, read_complex_dataset
from .provenance import redact_source_root, sha256_file

QUANTITIES = ("G", "G0", "Weiss", "hybridization", "Sigma", "spectral")


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return value


def _attributes(handle: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in handle.attrs.items()}


def _find_iteration_evidence(handle: h5py.File) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        path = f"/{name}" if name else "/"
        for key, value in obj.attrs.items():
            normalized = re.sub(r"[^a-z]", "", key.lower())
            if "iteration" in normalized:
                evidence.append({
                    "source_path": path,
                    "attribute": key,
                    "value": _json_value(value),
                })

    visit("", handle)
    handle.visititems(visit)
    return evidence


def _direct_quantity_roots(handle: h5py.File) -> dict[str, h5py.Group]:
    roots: dict[str, h5py.Group] = {}
    for quantity in QUANTITIES:
        obj = handle.get(quantity)
        if isinstance(obj, h5py.Group):
            roots[quantity] = obj
    return roots


def _numeric_payload(dataset: h5py.Dataset) -> np.ndarray | None:
    if dataset.name.endswith("/data"):
        return read_complex_dataset(dataset)
    dtype = dataset.dtype
    if dtype.names:
        names = {re.sub(r"[^a-z]", "", name.lower()): name for name in dtype.names}
        real_name = names.get("r") or names.get("real")
        imag_name = names.get("i") or names.get("imag")
        if real_name is not None and imag_name is not None:
            # Compound datasets are read as structured arrays with named
            # real/imaginary components.
            raw = dataset[()]
            result = raw[real_name].astype(np.float64) + 1j * raw[imag_name]
            return result
        return None
    if dtype.kind not in "iufc":
        return None
    return np.asarray(dataset[()])


def _payloads(group: h5py.Group) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        lower = name.lower()
        if "mesh" in lower:
            return
        values = _numeric_payload(obj)
        if values is not None:
            payloads.append({
                "source_path": f"{group.name.rstrip('/')}/{name}",
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "finite": bool(np.all(np.isfinite(values.real)) and np.all(
                    np.isfinite(values.imag)
                )),
                "values": values,
            })

    group.visititems(visit)
    return payloads


def inspect_triqs_file(file_path: str | Path, source_root: str | Path) -> dict[str, Any]:
    """Inspect one TRIQS-style file without modifying the source."""
    path = Path(file_path)
    root = Path(source_root)
    relative_path = path.relative_to(root).as_posix()
    sha256 = sha256_file(path)

    report: dict[str, Any] = {
        "relative_source_path": relative_path,
        "source_path": str(path.resolve()),
        "sha256": sha256,
        "status": "success",
        "file_attributes": {},
        "iteration_evidence": [],
        "metadata_iteration_values": [],
        "stored_iterations_by_quantity": {},
        "metadata_vs_stored_iterations": {},
        "quantities": {},
        "iteration_diagnostics": [],
        "convergence": {
            "status": "not_assessed",
            "last_iteration_is_converged": False,
            "reason": "No explicit metadata or log evidence was supplied.",
        },
        "errors": [],
    }

    try:
        with h5py.File(path, "r") as handle:
            report["file_attributes"] = _attributes(handle)
            report["iteration_evidence"] = _find_iteration_evidence(handle)

            roots = _direct_quantity_roots(handle)
            dmft = handle.get("dmft_output")
            if isinstance(dmft, h5py.Group):
                for quantity in QUANTITIES:
                    if any(re.fullmatch(rf"{quantity}-\d+", n) for n in dmft):
                        roots[quantity] = dmft
            for quantity, quantity_root in roots.items():
                iteration_groups: dict[str, h5py.Group] = {}
                for name, obj in quantity_root.items():
                    if isinstance(obj, h5py.Group):
                        flat_match = re.fullmatch(rf"{quantity}-(\d+)", name)
                        if quantity_root.name == "/dmft_output":
                            if flat_match:
                                iteration_groups[flat_match.group(1)] = obj
                        elif name.isdigit():
                            iteration_groups[name] = obj
                        elif re.search(r"(?:iteration[_-]?|iter[_-]?)(\d+)", name.lower()):
                            match = re.search(
                                r"(?:iteration[_-]?|iter[_-]?)(\d+)", name.lower()
                            )
                            iteration_groups[match.group(1)] = obj

                mesh_objects: list[str] = []
                for iteration_group in iteration_groups.values():
                    def collect_meshes(name: str, obj: h5py.Group | h5py.Dataset,
                                       mesh_objects=mesh_objects) -> None:
                        if "mesh" in name.lower() or "meshimfreq" in name.lower():
                            mesh_objects.append(obj.name)
                    iteration_group.visititems(collect_meshes)

                report["quantities"][quantity] = {
                    "source_path": quantity_root.name,
                    "attributes": _attributes(quantity_root),
                    "stored_iterations": sorted(
                        iteration_groups, key=lambda item: int(item)
                    ),
                    "channels": sorted({
                        name
                        for iteration_group in iteration_groups.values()
                        for name in iteration_group
                        if isinstance(iteration_group[name], h5py.Group)
                        and isinstance(iteration_group[name].get("data"), h5py.Dataset)
                    }),
                    "mesh_objects": sorted(set(mesh_objects)),
                }

                for left_name, right_name in zip(
                    sorted(iteration_groups, key=lambda item: int(item)),
                    sorted(iteration_groups, key=lambda item: int(item))[1:],
                ):
                    left = iteration_groups[left_name]
                    right = iteration_groups[right_name]
                    left_channels = {
                        name: obj
                        for name, obj in left.items()
                        if isinstance(obj, h5py.Group)
                    }
                    right_channels = {
                        name: obj
                        for name, obj in right.items()
                        if isinstance(obj, h5py.Group)
                    }
                    for channel in sorted(set(left_channels) & set(right_channels)):
                        left_payloads = _payloads(left_channels[channel])
                        right_payloads = _payloads(right_channels[channel])
                        for left_payload in left_payloads:
                            suffix = left_payload["source_path"].rsplit("/", 1)[-1]
                            for right_payload in right_payloads:
                                if right_payload["source_path"].rsplit("/", 1)[-1] != suffix:
                                    continue
                                if left_payload["shape"] != right_payload["shape"]:
                                    continue
                                left_values = left_payload["values"]
                                right_values = right_payload["values"]
                                report["iteration_diagnostics"].append({
                                    "role": quantity,
                                    "channel": channel,
                                    "from_iteration": left_name,
                                    "to_iteration": right_name,
                                    "left_source_path": left_payload["source_path"],
                                    "right_source_path": right_payload["source_path"],
                                    **difference_metrics(left_values, right_values),
                                })

            metadata_values = [
                item["value"]
                for item in report["iteration_evidence"]
                if item["attribute"].lower() in {"iterations", "iteration", "n_iterations"}
            ]
            report["metadata_iteration_values"] = metadata_values
            report["stored_iterations_by_quantity"] = {
                quantity: detail["stored_iterations"]
                for quantity, detail in report["quantities"].items()
            }
            for quantity, stored in report["stored_iterations_by_quantity"].items():
                report["metadata_vs_stored_iterations"][quantity] = {
                    "metadata_values": metadata_values,
                    "stored_count": len(stored),
                    "stored_iterations": stored,
                    "interpretation": (
                        "not_compared_to_metadata"
                        if not metadata_values
                        else "metadata_count_and_stored_iteration_range_are_recorded_separately"
                    ),
                }

            if report["iteration_evidence"]:
                convergence_keys = [
                    item for item in report["iteration_evidence"]
                    if "converg" in item["attribute"].lower()
                ]
                if convergence_keys:
                    report["convergence"] = {
                        "status": "evidence_present_not_evaluated",
                        "last_iteration_is_converged": False,
                        "reason": "Convergence evidence was found, but no repository threshold or independent validation is applied.",
                        "evidence": convergence_keys,
                    }
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        report["status"] = "failed"
        report["errors"].append({
            "error_type": type(error).__name__,
            "error": str(error),
        })

    # This JSON is shared for review; absolute source roots are not needed.
    return redact_source_root(report, root.resolve())

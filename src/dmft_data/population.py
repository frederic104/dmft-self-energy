"""Read-only population summaries for normalized compatibility datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_attribute_value(item) for item in value.tolist()]
    return value


def _labels(value: Any) -> list[str]:
    value = _attribute_value(value)
    if value in (None, "", "unknown"):
        return ["unknown"]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _array_digest(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset[()])
    digest = hashlib.sha256()
    digest.update(np.asarray(values.dtype.str).tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def summarize_compatibility_canonical(
    canonical_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Summarize a canonical compatibility HDF5 without copying array payloads."""
    canonical = Path(canonical_path)
    output = Path(output_path)
    if not canonical.is_file():
        report = {
            "status": "missing_input",
            "canonical_path": str(canonical),
            "record_count": 0,
            "limitations": ["Compatibility canonical HDF5 was not present in this workspace."],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    run_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    iteration_counts: Counter[str] = Counter()
    spin_counts: Counter[str] = Counter()
    orbital_counts: Counter[str] = Counter()
    channel_combinations: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    beta_or_metadata: Counter[str] = Counter()
    mesh_shape_counts: Counter[str] = Counter()
    mesh_signature_counts: Counter[str] = Counter()
    source_sha_counts: Counter[str] = Counter()
    source_path_counts: Counter[str] = Counter()
    warmup_counts: Counter[str] = Counter()
    exact_identity_counts: Counter[str] = Counter()
    payload_mesh_counts: Counter[str] = Counter()
    mesh_positive_imag_counts: Counter[str] = Counter()
    mesh_negative_imag_counts: Counter[str] = Counter()
    mesh_real_only_counts: Counter[str] = Counter()
    warnings: list[dict[str, Any]] = []

    with h5py.File(canonical, "r") as handle:
        runs = handle.get("runs")
        if not isinstance(runs, h5py.Group):
            raise KeyError("Compatibility canonical HDF5 missing /runs")

        for name in sorted(runs.keys()):
            group = runs[name]
            if not isinstance(group, h5py.Group):
                continue
            run_id = str(_attribute_value(group.attrs.get("run_id", "unknown")))
            state_id = str(_attribute_value(
                group.attrs.get("physical_state_id", "unknown")
            ))
            role = str(_attribute_value(group.attrs.get("role", "unknown")))
            iteration = str(_attribute_value(group.attrs.get("iteration", "unknown")))
            source_path = str(_attribute_value(
                group.attrs.get("source_path", "unknown")
            ))
            source_sha = str(_attribute_value(
                group.attrs.get("source_sha256", "unknown")
            ))
            warmup = bool(_attribute_value(group.attrs.get("warmup", False)))
            spin_labels = _labels(group.attrs.get("spin_labels"))
            orbital_labels = _labels(group.attrs.get("orbital_labels"))

            run_counts[run_id] += 1
            state_counts[state_id] += 1
            role_counts[role] += 1
            iteration_counts[iteration] += 1
            warmup_counts["warmup" if warmup else "not_warmup"] += 1
            source_sha_counts[source_sha] += 1
            source_path_counts[source_path] += 1
            for spin in spin_labels:
                spin_counts[spin] += 1
            for orbital in orbital_labels:
                orbital_counts[orbital] += 1
            channel_combinations[
                json.dumps([spin_labels, orbital_labels], sort_keys=True)
            ] += 1

            temperature = str(_attribute_value(
                group.attrs.get("metadata_temperature", "unknown")
            ))
            beta_or_metadata[temperature] += 1

            mesh = np.asarray(group["matsubara_mesh"][()])
            mesh_shape_counts[f"{mesh.shape}"] += 1
            mesh_digest = hashlib.sha256()
            mesh_digest.update(np.asarray(mesh.dtype.str).tobytes())
            mesh_digest.update(mesh.tobytes(order="C"))
            mesh_signature_counts[mesh_digest.hexdigest()] += 1
            if np.iscomplexobj(mesh):
                positive = int(np.count_nonzero(np.imag(mesh) > 0))
                negative = int(np.count_nonzero(np.imag(mesh) < 0))
                real_only = int(np.count_nonzero(np.imag(mesh) == 0))
            else:
                positive = 0
                negative = 0
                real_only = mesh.size
            mesh_positive_imag_counts[str(positive)] += 1
            mesh_negative_imag_counts[str(negative)] += 1
            mesh_real_only_counts[str(real_only)] += 1

            values = np.asarray(group["values"][()])
            payload_digest = _array_digest(group["values"])
            payload_mesh_counts[f"{payload_digest}:{mesh_digest.hexdigest()}"] += 1

            exact_identity = json.dumps(
                {
                    "physical_state_id": state_id,
                    "run_id": run_id,
                    "iteration": iteration,
                    "role": role,
                    "source_path": source_path,
                    "spin_labels": spin_labels,
                    "orbital_labels": orbital_labels,
                },
                sort_keys=True,
            )
            exact_identity_counts[exact_identity] += 1

    duplicate_exact = sum(count - 1 for count in exact_identity_counts.values() if count > 1)
    duplicate_payload_mesh = sum(
        count - 1 for count in payload_mesh_counts.values() if count > 1
    )

    report: dict[str, Any] = {
        "status": "passed",
        "canonical_path": str(canonical),
        "record_count": sum(run_counts.values()),
        "independent_physical_state_count": len(state_counts),
        "physical_state_ids": sorted(state_counts),
        "run_count": len(run_counts),
        "run_record_counts": dict(sorted(run_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "iteration_counts": dict(sorted(iteration_counts.items())),
        "spin_label_counts": dict(sorted(spin_counts.items())),
        "orbital_label_counts": dict(sorted(orbital_counts.items())),
        "channel_combination_counts": dict(sorted(channel_combinations.items())),
        "temperature_metadata_counts": dict(sorted(beta_or_metadata.items())),
        "mesh_shape_counts": dict(sorted(mesh_shape_counts.items())),
        "mesh_signature_count": len(mesh_signature_counts),
        "mesh_positive_imag_count_distribution": dict(sorted(
            mesh_positive_imag_counts.items(), key=lambda item: int(item[0])
        )),
        "mesh_negative_imag_count_distribution": dict(sorted(
            mesh_negative_imag_counts.items(), key=lambda item: int(item[0])
        )),
        "mesh_real_only_count_distribution": dict(sorted(
            mesh_real_only_counts.items(), key=lambda item: int(item[0])
        )),
        "warmup_counts": dict(sorted(warmup_counts.items())),
        "source_sha256_unique_count": len(source_sha_counts),
        "source_sha256_record_counts": dict(sorted(source_sha_counts.items())),
        "source_path_unique_count": len(source_path_counts),
        "source_path_record_counts": dict(sorted(source_path_counts.items())),
        "duplicate_exact_identity_count": duplicate_exact,
        "equivalent_payload_and_mesh_duplicate_count": duplicate_payload_mesh,
        "record_axis_definition": {
            "unit": "one physical-quantity array dataset per record",
            "run": "calculation run identifier",
            "iteration": "stored iteration identifier",
            "channel_labels": "spin/orbital metadata attached to the array",
            "frequency": "matsubara_mesh and frequency_axis identify array frequency axis",
        },
        "limitations": [
            "Independent physical-state count is a state-ID grouping count, "
            "not evidence that the registry is physically complete.",
            "Final/last equality and payload duplicates are not convergence evidence.",
            "Beta values are not inferred as physical metadata from this report.",
        ],
        "warnings": warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

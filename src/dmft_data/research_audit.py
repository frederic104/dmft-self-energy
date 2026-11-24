"""Fail-closed provenance, target, replica, and split audits for raw DMFT runs.

The outputs describe evidence available in HDF5 storage and supplied registries.
They never infer physical convergence or independent physical states from paths,
warmup settings, iterations, spin channels, or orbital channels.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py

from .hdf import physical_blocks
from .provenance import sha256_file as _sha256

UNKNOWN = "unknown"
SCHEMA_VERSION = 1


def _load_object(path: Path | None, label: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _load_diagnostics(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"Run diagnostics must be a JSON list: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(
            row.get("relative_source_path"), str
        ):
            raise TypeError("Each run-diagnostics row requires relative_source_path")
        rows[row["relative_source_path"]] = row
    return rows


def _scalar(value: Any) -> str | int | float | bool:
    if value is None:
        return UNKNOWN
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _run_metadata(metadata: dict[str, Any], relative: str) -> dict[str, Any]:
    runs = metadata.get("runs", metadata)
    value = runs.get(relative, {}) if isinstance(runs, dict) else {}
    return value if isinstance(value, dict) else {}


def _final_block_names(dmft: h5py.Group, quantity: str) -> list[str]:
    """List only BlockGf-like groups with a ``data`` dataset."""
    final = dmft.get(f"{quantity}_iw")
    if not isinstance(final, h5py.Group):
        return []
    return list(physical_blocks(final))


def _storage_schema(path: Path) -> dict[str, Any]:
    """Capture directly observable final-array structure, never physical meaning."""
    try:
        with h5py.File(path, "r") as handle:
            dmft = handle.get("dmft_output")
            if not isinstance(dmft, h5py.Group):
                return {"status": "missing_dmft_output", "final_blocks": {}}
            return {
                "status": "observed",
                "final_blocks": {
                    "G_iw": _final_block_names(dmft, "G"),
                    "Sigma_iw": _final_block_names(dmft, "Sigma"),
                },
            }
    except OSError as error:
        return {"status": "unreadable", "error": str(error), "final_blocks": {}}


def _state_metadata(metadata: dict[str, Any], state_id: str) -> dict[str, Any]:
    states = metadata.get("physical_states", {})
    value = states.get(state_id, {}) if isinstance(states, dict) else {}
    return value if isinstance(value, dict) else {}


def build_physical_state_manifest(
    source_root: str | Path,
    output_path: str | Path,
    *,
    diagnostics_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write run provenance without claiming that candidate groups are states."""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    diagnostics = _load_diagnostics(
        Path(diagnostics_path).resolve() if diagnostics_path else None
    )
    metadata = _load_object(
        Path(metadata_path).resolve() if metadata_path else None,
        "Physical-state metadata",
    )
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    records = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        diagnostic = diagnostics.get(relative, {})
        supplied = _run_metadata(metadata, relative)
        state_id = str(supplied.get("physical_state_id", UNKNOWN))
        state = _state_metadata(metadata, state_id)
        material = supplied.get("material", diagnostic.get("material", UNKNOWN))
        beta = supplied.get(
            "beta", diagnostic.get("mesh_metadata", {}).get("beta", UNKNOWN)
        )
        records.append({
            "relative_source_path": relative,
            "source_sha256": _sha256(path),
            "file_size_bytes": path.stat().st_size,
            "hdf5_storage_schema": _storage_schema(path),
            "material": _scalar(material),
            "beta": _scalar(beta),
            "nwarmup": _scalar(diagnostic.get("nwarmup")),
            "parameter_variation_candidate": bool(
                diagnostic.get("parameter_variation_candidate", False)
            ),
            "storage_status": diagnostic.get("run_status", "not_diagnosed"),
            "convergence_status": "not_assessed",
            "physical_state_id": state_id,
            "physical_state_id_source": (
                "supplied_metadata" if state_id != UNKNOWN else "unresolved"
            ),
            "state_metadata": {
                key: _scalar(state.get(key, supplied.get(key)))
                for key in ("temperature", "U", "JH", "filling", "structure",
                            "double_counting", "solver", "seed")
            },
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(root),
        "record_count": len(records),
        "records": records,
        "limitations": [
            "A physical_state_id is accepted only from supplied metadata.",
            ("Path, beta, warmup, iteration, spin, and orbital information do not "
             "establish an independent physical state."),
            "Storage diagnostics do not establish DMFT convergence.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _registry_record(registry: dict[str, Any], relative: str) -> dict[str, Any]:
    runs = registry.get("runs", registry)
    value = runs.get(relative, {}) if isinstance(runs, dict) else {}
    return value if isinstance(value, dict) else {}


def build_target_registry(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    convergence_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select primary targets only from explicitly verified external evidence."""
    manifest = _load_object(Path(manifest_path), "Physical-state manifest")
    registry = _load_object(
        (
            Path(convergence_registry_path).resolve()
            if convergence_registry_path
            else None
        ),
        "Convergence registry",
    )
    records = []
    for run in manifest.get("records", []):
        relative = run["relative_source_path"]
        evidence = _registry_record(registry, relative)
        verified = (
            evidence.get("status") == "verified_converged"
            and isinstance(evidence.get("evidence_source"), str)
            and bool(evidence["evidence_source"].strip())
            and type(evidence.get("selected_iteration")) is int
            and evidence.get("source_sha256") == run.get("source_sha256")
            and bool(evidence.get("source_sha256"))
            and isinstance(evidence.get("selected_quantity_paths"), dict)
            and isinstance(evidence["selected_quantity_paths"].get("Sigma_iw"), str)
            and evidence["selected_quantity_paths"]["Sigma_iw"].startswith("/")
        )
        records.append({
            "relative_source_path": relative,
            "physical_state_id": run["physical_state_id"],
            "source_sha256": run.get("source_sha256"),
            "storage_status": run["storage_status"],
            "selected_iteration": evidence.get("selected_iteration"),
            "selected_quantity_paths": evidence.get("selected_quantity_paths", {}),
            "convergence_status": (
                "verified_converged" if verified else "not_assessed"
            ),
            "evidence_source": evidence.get("evidence_source"),
            "primary_target_candidate": verified,
            "reason": (
                "explicit_external_convergence_registry" if verified
                else "no_explicit_verified_convergence_record"
            ),
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "primary_target_count": sum(r["primary_target_candidate"] for r in records),
        "records": records,
        "limitations": [
            "An entry is not verified merely because final and iteration arrays match.",
            "Registry evidence is recorded, not independently validated by this tool.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def audit_replicas(
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Report run replicas and metadata gaps without declaring independence."""
    manifest = _load_object(Path(manifest_path), "Physical-state manifest")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved = []
    for record in manifest.get("records", []):
        state_id = record.get("physical_state_id", UNKNOWN)
        if state_id == UNKNOWN:
            unresolved.append(record["relative_source_path"])
            continue
        groups[str(state_id)].append(record)
    result_groups = []
    for state_id, members in sorted(groups.items()):
        result_groups.append({
            "physical_state_id": state_id,
            "member_count": len(members),
            "relative_source_paths": sorted(
                r["relative_source_path"] for r in members
            ),
            "nwarmup_values": sorted({str(r["nwarmup"]) for r in members}),
            "storage_statuses": sorted({str(r["storage_status"]) for r in members}),
            "parameter_variation_candidate_count": sum(
                bool(r["parameter_variation_candidate"]) for r in members
            ),
            "interpretation": "replica_group_from_explicit_physical_state_id",
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "replica_groups": result_groups,
        "unresolved_run_count": len(unresolved),
        "unresolved_relative_source_paths": sorted(unresolved),
        "limitations": [
            ("A replica group does not prove member arrays are statistically "
             "independent."),
            "Runs without supplied physical_state_id are intentionally not grouped.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def create_blocked_splits(
    manifest_path: str | Path,
    target_registry_path: str | Path,
    output_path: str | Path,
    *,
    minimum_states: int = 3,
) -> dict[str, Any]:
    """Create leave-one-material-out folds from verified explicit states only."""
    manifest = _load_object(Path(manifest_path), "Physical-state manifest")
    targets = _load_object(Path(target_registry_path), "Target registry")
    accepted = {
        entry["relative_source_path"] for entry in targets.get("records", [])
        if entry.get("primary_target_candidate") is True
    }
    states: dict[str, dict[str, Any]] = {}
    for record in manifest.get("records", []):
        if record["relative_source_path"] not in accepted:
            continue
        state_id = record.get("physical_state_id", UNKNOWN)
        material = record.get("material", UNKNOWN)
        if state_id == UNKNOWN or material == UNKNOWN:
            continue
        states.setdefault(str(state_id), record)
    if len(states) < minimum_states:
        result = {
            "status": "insufficient_verified_physical_states",
            "verified_physical_state_count": len(states),
            "minimum_physical_states": minimum_states,
            "folds": [],
            "reason": "No split is created from unresolved or unverified targets.",
        }
    else:
        by_material: dict[str, list[str]] = defaultdict(list)
        for state_id, record in states.items():
            by_material[str(record["material"])].append(state_id)
        folds = [
            {
                "held_out_material": material,
                "held_out_state_ids": sorted(held_out),
                "train_state_ids": sorted(set(states) - set(held_out)),
            }
            for material, held_out in sorted(by_material.items())
            if set(states) - set(held_out)
        ]
        result = {
            "status": (
                "created"
                if len(by_material) >= 2 and folds
                else "insufficient_material_blocks"
            ),
            "verified_physical_state_count": len(states),
            "minimum_physical_states": minimum_states,
            "folds": folds if len(by_material) >= 2 else [],
            "limitations": [
                ("Temperature- and parameter-blocked folds require separately "
                 "verified metadata."),
            ],
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result

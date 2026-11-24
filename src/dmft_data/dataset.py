"""Construction of verified, state-aware DMFT research datasets."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .extraction import software_versions

LOGGER = logging.getLogger(__name__)


class InsufficientPhysicalStatesError(ValueError):
    """Raised when a caller requires splits without enough independent states."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _attribute(handle: h5py.File, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in handle.attrs:
            return handle.attrs[key]
    return None


def _array_attribute(dataset: h5py.Dataset, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in dataset.attrs:
            return dataset.attrs[key]
    return None


def _explicit_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "converged", "verified"}
    return False


def _is_verified_converged(
    source: h5py.File,
    dataset: h5py.Dataset,
    run_id: str,
    state_id: str,
    relative_path: str,
    verification_records: dict[str, Any],
) -> bool:
    """Accept convergence only from an explicit, named verification source."""
    # A global boolean, state-level declaration, or unreviewed stored flag must
    # never certify every array in a collection. Bind evidence to source bytes
    # and exact original array paths, as in the newer paired learning workflow.
    records = verification_records.get("runs", verification_records)
    for key in (relative_path, run_id):
        record = records.get(key, {})
        if not isinstance(record, dict):
            continue
        if (
            record.get("status") == "verified_converged"
            and isinstance(record.get("evidence_source"), str)
            and bool(record["evidence_source"].strip())
            and record.get("source_sha256") == source.attrs.get("source_sha256")
            and bool(record.get("source_sha256"))
            and isinstance(record.get("verified_source_paths"), list)
            and dataset.attrs.get("source_path") in record.get("verified_source_paths", [])
        ):
            return True
    return False


def _is_explicit_warmup(dataset: h5py.Dataset, source_path: str) -> bool:
    for key in ("warmup", "is_warmup", "warmup_iteration"):
        if key in dataset.attrs and _explicit_bool(dataset.attrs[key]):
            return True
    for key in ("iteration_type", "iteration_kind", "phase"):
        if key in dataset.attrs:
            return str(dataset.attrs[key]).strip().lower() == "warmup"
    source_name = source_path.lower()
    return "/warmup/" in source_name or "/warmup_" in source_name


def _load_verification_records(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid convergence verification JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Convergence verification must be a JSON object: {path}")
    return value


def _read_metadata_records(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Physical-state metadata must be a JSON object: {path}")
    return value


def _metadata_value(
    source: h5py.File,
    state_id: str,
    run_id: str,
    metadata_records: dict[str, Any],
    keys: Iterable[str],
) -> Any:
    for key in keys:
        if key in source.attrs:
            return source.attrs[key]
    for record in (
        metadata_records.get(state_id, {}),
        metadata_records.get(run_id, {}),
        metadata_records,
    ):
        if not isinstance(record, dict):
            continue
        for key in keys:
            if key in record:
                return record[key]
    return None


def _safe_scalar(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return _json(value)
    return str(value)


def _state_metadata(
    source: h5py.File,
    state_id: str,
    run_id: str,
    metadata_records: dict[str, Any],
) -> dict[str, str]:
    return {
        "material": _safe_scalar(
            _metadata_value(
                source, state_id, run_id, metadata_records,
                ("material", "material_name", "compound"),
            )
        ),
        "temperature": _safe_scalar(
            _metadata_value(
                source, state_id, run_id, metadata_records,
                ("temperature", "temperature_K", "T", "T_K"),
            )
        ),
        "parameter_block": _safe_scalar(
            _metadata_value(
                source, state_id, run_id, metadata_records,
                ("parameter_block",),
            )
        ),
        "U": _safe_scalar(
            _metadata_value(source, state_id, run_id, metadata_records,
                            ("U", "U_eV", "Hubbard_U"))
        ),
        "JH": _safe_scalar(
            _metadata_value(source, state_id, run_id, metadata_records,
                            ("JH", "J_H", "J_h", "Hund_J"))
        ),
        "filling": _safe_scalar(
            _metadata_value(source, state_id, run_id, metadata_records,
                            ("filling", "density", "occupation"))
        ),
    }


def _group_tag(path: Path) -> str:
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def _write_canonical(
    output: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    state_ids = sorted({record["physical_state_id"] for record in records})
    with h5py.File(temporary, "w") as handle:
        handle.attrs["record_format"] = "state_run_iteration_channel_preserving"
        handle.attrs["canonical"] = True
        handle.attrs["software_versions"] = _json(software_versions())
        handle.attrs["manifest"] = _json(manifest)
        states_group = handle.create_group("physical_states")
        for state_id in state_ids:
            state_group = states_group.create_group(state_id)
            state_group.attrs["physical_state_id"] = state_id
            state_records = [r for r in records if r["physical_state_id"] == state_id]
            state_group.attrs["run_count"] = len({r["run_id"] for r in state_records})
        runs_group = handle.create_group("runs")
        for index, record in enumerate(records):
            run_id = record["run_id"]
            iteration = record["iteration"]
            role = record["role"]
            source_path = record["source_path"]
            group_name = f"record_{index:06d}"
            group = runs_group.create_group(group_name)
            group.attrs["physical_state_id"] = record["physical_state_id"]
            group.attrs["run_id"] = run_id
            group.attrs["iteration"] = iteration
            group.attrs["role"] = role
            group.attrs["source_path"] = source_path
            group.attrs["source_relative_path"] = record["source_relative_path"]
            group.attrs["source_sha256"] = record["source_sha256"]
            group.attrs["warmup"] = record["warmup"]
            group.attrs["spin_labels"] = record["spin_labels"] or "unknown"
            group.attrs["orbital_labels"] = record["orbital_labels"] or "unknown"
            for key, value in record["metadata"].items():
                group.attrs[f"metadata_{key}"] = value
            values = np.asarray(record["values"])
            group.create_dataset("values", data=values)
            mesh = np.asarray(record["frequency_mesh"])
            mesh_dataset = group.create_dataset("matsubara_mesh", data=mesh)
            mesh_dataset.attrs["source_path"] = record["frequency_mesh_source"]
            group.attrs["frequency_axis"] = record["frequency_axis"]
    temporary.replace(output)
    return {
        "path": str(output),
        "physical_state_count": len(state_ids),
        "physical_state_ids": state_ids,
        "record_count": len(records),
    }


def _derive_reduced_window(
    canonical_path: Path,
    output: Path,
    window_name: str,
    window_config: dict[str, Any],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with h5py.File(canonical_path, "r") as source, h5py.File(temporary, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["derived_from"] = str(canonical_path)
        target.attrs["window_name"] = window_name
        target.attrs["window_config"] = _json(window_config)
        target_states = target.create_group("physical_states")
        source_states = source["physical_states"]
        for name in source_states:
            source_states.copy(name, target_states, name=name)
        target_runs = target.create_group("runs")
        for name in source["runs"]:
            source_group = source["runs"][name]
            mesh = np.asarray(source_group["matsubara_mesh"])
            axis = int(source_group.attrs["frequency_axis"])
            start = window_config.get("start")
            stop = window_config.get("stop")
            count = window_config.get("count")
            positive_frequency = bool(window_config.get("positive_frequency", False))
            if positive_frequency:
                if start is not None or stop is not None:
                    raise ValueError(
                        f"Reduced window {window_name!r} cannot combine "
                        "positive_frequency with start/stop"
                    )
                positive_indices = np.flatnonzero(np.imag(mesh) > 0)
                if not np.all(np.diff(positive_indices) >= 0):
                    raise ValueError(
                        f"Reduced window {window_name!r} has unsorted positive "
                        "frequency indices"
                    )
                if not isinstance(count, int) or count < 1:
                    raise ValueError(
                        f"Reduced window {window_name!r} requires a positive "
                        "integer count"
                    )
                if positive_indices.size < count:
                    raise ValueError(
                        f"Reduced window {window_name!r} requests {count} positive "
                        f"frequencies but mesh contains only {positive_indices.size}"
                    )
                selected = positive_indices[:count]
            elif start is not None or stop is not None:
                selected = np.arange(mesh.size)[slice(start, stop)]
            elif isinstance(count, int) and count >= 1:
                if mesh.size < count:
                    raise ValueError(
                        f"Reduced window {window_name!r} requests {count} "
                        f"frequencies but mesh contains only {mesh.size}"
                    )
                selected = np.arange(count)
            else:
                raise ValueError(
                    f"Reduced window {window_name!r} requires start/stop or count"
                )
            if selected.size == 0:
                raise ValueError(f"Reduced window {window_name!r} selects no frequencies")
            target_group = target_runs.create_group(name)
            for key, value in source_group.attrs.items():
                target_group.attrs[key] = value
            values = np.asarray(source_group["values"])
            slicer = [slice(None)] * values.ndim
            slicer[axis] = selected
            target_group.create_dataset("values", data=values[tuple(slicer)])
            target_mesh = target_group.create_dataset(
                "matsubara_mesh", data=mesh[selected]
            )
            target_mesh.attrs["source_path"] = (
                source_group["matsubara_mesh"].attrs["source_path"]
            )
            target_group.attrs["window_source_indices"] = selected
    temporary.replace(output)
    return {"name": window_name, "path": str(output), "config": window_config}


def _blocked_splits(
    records: list[dict[str, Any]],
    minimum_states: int,
) -> dict[str, Any]:
    state_ids = sorted({record["physical_state_id"] for record in records})
    state_metadata: dict[str, dict[str, str]] = {}
    for state_id in state_ids:
        record = next(r for r in records if r["physical_state_id"] == state_id)
        state_metadata[state_id] = record["metadata"]

    dimensions = ("material", "temperature", "parameter_block")
    split_result: dict[str, Any] = {
        "minimum_physical_states": minimum_states,
        "independent_physical_state_count": len(state_ids),
        "independent_physical_state_ids": state_ids,
        "status": "created",
        "limitations": [],
        "splits": {},
    }
    if len(state_ids) < minimum_states:
        split_result["status"] = "insufficient_physical_states"
        split_result["limitations"].append(
            f"Only {len(state_ids)} independent physical states are available; "
            f"{minimum_states} are required."
        )
        return split_result

    for dimension in dimensions:
        groups: dict[str, list[str]] = {}
        for state_id in state_ids:
            value = state_metadata[state_id].get(dimension)
            if value in (None, "", "unknown"):
                continue
            groups.setdefault(value, []).append(state_id)
        if len(groups) < 2:
            split_result["splits"][dimension] = {
                "status": "insufficient_block_metadata",
                "exact_state_ids_by_group": groups,
                "folds": [],
            }
            continue
        folds = []
        for held_out, held_ids in sorted(groups.items()):
            train_ids = sorted(set(state_ids) - set(held_ids))
            if not train_ids:
                continue
            folds.append({
                "held_out_group": held_out,
                "train_state_ids": train_ids,
                "held_out_state_ids": sorted(held_ids),
            })
        split_result["splits"][dimension] = {
            "status": "created" if folds else "insufficient_block_metadata",
            "exact_state_ids_by_group": groups,
            "folds": folds,
        }
    return split_result


def _collect_records(
    normalized_dir: Path,
    metadata_records: dict[str, Any],
    verification_records: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    converged: list[dict[str, Any]] = []
    intermediate: list[dict[str, Any]] = []
    compatibility: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not normalized_dir.is_dir():
        return converged, intermediate, compatibility, [{
            "code": "normalized_data_directory_missing",
            "path": str(normalized_dir),
        }]

    for source_path in sorted(normalized_dir.rglob("*.h5")):
        try:
            with h5py.File(source_path, "r") as source:
                relative_path = source_path.relative_to(normalized_dir).as_posix()
                run_id = str(source.attrs["run_id"])
                state_id = str(source.attrs["physical_state_id"])
                run_explicit = _explicit_bool(source.attrs.get("run_id_explicit", True))
                state_explicit = _explicit_bool(
                    source.attrs.get("physical_state_id_explicit", True)
                )
                if not run_explicit:
                    warnings.append({
                        "code": "run_id_not_explicit",
                        "source_relative_path": relative_path,
                        "run_id": run_id,
                    })
                if not state_explicit or state_id in {"unknown", "none", ""}:
                    warnings.append({
                        "code": "physical_state_id_not_explicit",
                        "source_relative_path": relative_path,
                        "physical_state_id": state_id,
                    })
                    continue
                metadata = _state_metadata(
                    source, state_id, run_id, metadata_records
                )
                sha256 = str(source.attrs["source_sha256"])
                arrays = source["run"]["arrays"]
                for name, dataset in arrays.items():
                    if not name.startswith("array_") or not isinstance(dataset, h5py.Dataset):
                        continue
                    role = str(dataset.attrs["role"])
                    if role not in {
                        "impurity_self_energy",
                        "lattice_self_energy",
                        "dft_derived_g0",
                        "dmft_weiss",
                        "aim_hybridization",
                        "impurity_green",
                        "lattice_green",
                        "green_function",
                        "self_energy",
                    }:
                        continue
                    index = name.split("_")[-1]
                    mesh_name = f"mesh_{index}"
                    if mesh_name not in arrays:
                        warnings.append({
                            "code": "frequency_mesh_missing",
                            "source_relative_path": relative_path,
                            "source_path": str(dataset.attrs["source_path"]),
                        })
                        continue
                    mesh = np.asarray(arrays[mesh_name][()])
                    frequency_axis = int(dataset.attrs.get("frequency_axis", -1))
                    values = np.asarray(dataset[()])
                    if values.ndim == 0 or values.shape[
                        frequency_axis if frequency_axis >= 0 else values.ndim - 1
                    ] != mesh.size:
                        warnings.append({
                            "code": "frequency_mesh_shape_mismatch",
                            "source_relative_path": relative_path,
                            "source_path": str(dataset.attrs["source_path"]),
                        })
                        continue
                    spin_labels = dataset.attrs.get("spin_labels", None)
                    orbital_labels = dataset.attrs.get("orbital_labels", None)
                    if isinstance(spin_labels, np.ndarray):
                        spin_labels = spin_labels.tolist()
                    if isinstance(orbital_labels, np.ndarray):
                        orbital_labels = orbital_labels.tolist()
                    iteration = str(dataset.attrs.get("iteration", "unknown"))
                    warmup = _is_explicit_warmup(
                        dataset, str(dataset.attrs["source_path"])
                    )
                    record = {
                        "physical_state_id": state_id,
                        "run_id": run_id,
                        "iteration": iteration,
                        "role": role,
                        "source_path": str(dataset.attrs["source_path"]),
                        "source_relative_path": relative_path,
                        "source_sha256": sha256,
                        "frequency_mesh": mesh,
                        "frequency_mesh_source": str(
                            arrays[mesh_name].attrs["source_path"]
                        ),
                        "frequency_axis": frequency_axis,
                        "spin_labels": spin_labels,
                        "orbital_labels": orbital_labels,
                        "warmup": warmup,
                        "metadata": metadata,
                        "values": values,
                    }
                    compatibility.append(record)
                    if _is_verified_converged(
                        source,
                        dataset,
                        run_id,
                        state_id,
                        relative_path,
                        verification_records,
                    ):
                        converged.append(record)
                    else:
                        intermediate.append(record)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            warnings.append({
                "code": "normalized_file_failed",
                "source_relative_path": source_path.as_posix(),
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return converged, intermediate, compatibility, warnings


def build_research_datasets(
    config_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build canonical, auxiliary, derived, and blocked-split research outputs."""
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

    paths = config["paths"]
    processed_dir = root / paths["processed_dir"]
    report_dir = root / paths["report_dir"]
    normalized_dir = processed_dir / "arrays"
    dataset_config = config.get("dataset", {})

    metadata_path_value = dataset_config.get(
        "physical_state_metadata", "data/metadata/physical_states.json"
    )
    metadata_path = root / metadata_path_value
    verification_path_value = dataset_config.get(
        "convergence_verification", "data/metadata/convergence_verification.json"
    )
    verification_path = root / verification_path_value
    metadata_records = _read_metadata_records(metadata_path)
    verification_records = _load_verification_records(verification_path)

    converged, intermediate, compatibility, warnings = _collect_records(
        normalized_dir, metadata_records, verification_records
    )
    self_energy_roles = {
        "impurity_self_energy",
        "lattice_self_energy",
        "self_energy",
    }
    baseline_records = [
        record for record in converged if record["role"] in self_energy_roles
    ]
    physics_records = [
        record
        for record in converged
        if record["role"] in self_energy_roles
        or record["role"] in {"dft_derived_g0", "dmft_weiss", "aim_hybridization"}
    ]
    physics_records = [record for record in physics_records if not record["warmup"]]
    compatibility_records = compatibility

    output_root = processed_dir / "research_datasets"
    baseline_dir = output_root / "A_deep_dmft_reproduction"
    physics_dir = output_root / "B_physics_constrained"
    compatibility_dir = output_root / "compatibility_deep_dmft"
    auxiliary_dir = output_root / "auxiliary"
    derived_dir = output_root / "derived"
    report_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = {
        "project": config.get("project", {}),
        "source_normalized_dir": str(normalized_dir.resolve()),
        "metadata_file": str(metadata_path.resolve()) if metadata_path.is_file() else None,
        "convergence_verification_file": (
            str(verification_path.resolve()) if verification_path.is_file() else None
        ),
        "software_versions": software_versions(),
        "convergence_policy": "explicit verification only",
        "warmup_policy": "excluded from physics-constrained primary dataset",
    }
    baseline_info = _write_canonical(
        baseline_dir / "canonical.h5",
        baseline_records,
        {**base_manifest, "dataset": "A_deep_dmft_reproduction"},
    )
    physics_info = _write_canonical(
        physics_dir / "canonical.h5",
        physics_records,
        {**base_manifest, "dataset": "B_physics_constrained"},
    )
    compatibility_info = _write_canonical(
        compatibility_dir / "canonical.h5",
        compatibility_records,
        {
            **base_manifest,
            "dataset": "deep_dmft_original_setup_compatibility",
            "mode": "compatibility",
        },
    )
    auxiliary_info = _write_canonical(
        auxiliary_dir / "intermediate_iterations.h5",
        intermediate,
        {**base_manifest, "dataset": "auxiliary_intermediate_iterations"},
    )

    windows = dataset_config.get("frequency_windows", [])
    derived_outputs = []
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, dict) or "name" not in window:
                continue
            name = str(window["name"])
            safe_name = "".join(
                char if char.isalnum() or char in "_-" else "_" for char in name
            )
            window_config = {key: value for key, value in window.items() if key != "name"}
            for label, canonical in (
                ("deep_dmft", baseline_dir / "canonical.h5"),
                ("physics", physics_dir / "canonical.h5"),
                ("compatibility", compatibility_dir / "canonical.h5"),
            ):
                result = _derive_reduced_window(
                    canonical,
                    derived_dir / f"{label}_{safe_name}.h5",
                    name,
                    window_config,
                )
                derived_outputs.append(result)

    split_config = dataset_config.get("splits", {})
    if not isinstance(split_config, dict):
        split_config = {}
    minimum_states = int(split_config.get("minimum_physical_states", 3))
    split_result = _blocked_splits(physics_records, minimum_states)
    split_path = output_root / "blocked_splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(_json(split_result) + "\n", encoding="utf-8")

    report = {
        "status": "passed" if not warnings and baseline_records and physics_records else "limited",
        "baseline": baseline_info,
        "physics_constrained": physics_info,
        "compatibility": compatibility_info,
        "auxiliary": auxiliary_info,
        "derived": derived_outputs,
        "blocked_splits": split_result,
        "blocked_splits_path": str(split_path),
        "warnings": warnings,
        "limitations": [],
    }
    if not baseline_records:
        report["limitations"].append(
            "No explicitly verified converged self-energy targets were found."
        )
    if not physics_records:
        report["limitations"].append(
            "No physics-constrained converged records passed the warmup exclusion."
        )
    if split_result["status"] == "insufficient_physical_states":
        report["status"] = "insufficient_physical_states"

    report_path = report_dir / "research_dataset_build.json"
    report_path.write_text(_json(report) + "\n", encoding="utf-8")
    return report


# Public compatibility aliases for callers that prefer either naming.
construct_research_datasets = build_research_datasets
build_datasets = build_research_datasets

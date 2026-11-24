"""Prepare paired arrays from explicit run/path selections and review metadata."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..hdf import read_quantity
from ..provenance import sha256_file


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def present(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in {
        "",
        "unknown",
        "none",
        "review-pending",
        "unresolved",
    }


def catalog(source: Path, output: Path) -> dict:
    """Inspect names and small mesh metadata; never read physical data payloads."""
    if not source.is_dir():
        raise FileNotFoundError(source)
    records, failures = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
            continue
        relative = path.relative_to(source).as_posix()
        try:
            with h5py.File(path, "r") as handle:
                quantities, beta_evidence = {}, []

                def visit(
                    name, obj, quantities=quantities, beta_evidence=beta_evidence
                ):
                    if isinstance(obj, h5py.Group) and isinstance(
                        obj.get("data"), h5py.Dataset
                    ):
                        quantities["/" + name] = {
                            "shape": list(obj["data"].shape),
                            "dtype": str(obj["data"].dtype),
                        }
                    if (
                        isinstance(obj, h5py.Dataset)
                        and name.endswith("/mesh/domain/beta")
                        and obj.shape == ()
                    ):
                        beta_evidence.append(
                            {"path": "/" + name, "value": float(obj[()])}
                        )

                handle.visititems(visit)
                records.append(
                    {
                        "relative_source_path": relative,
                        "material_candidate": relative.split("/")[0],
                        "material": relative.split("/")[0],
                        "material_evidence": "source_directory_name_only",
                        "nwarmup": (
                            int(match.group(1))
                            if (match := re.search(r"nwarmup(\d+)", path.stem))
                            else None
                        ),
                        "parameter_variation_candidate": "/beta-variation/"
                        in f"/{relative}",
                        "beta": (
                            beta_evidence[0]["value"]
                            if beta_evidence
                            and len({b["value"] for b in beta_evidence}) == 1
                            else None
                        ),
                        "quantity_blocks": quantities,
                        "beta_evidence": beta_evidence,
                        "physical_state_id": None,
                        "replica_group_id": None,
                        "input_path": None,
                        "target_path": None,
                        "input_semantics": "unresolved",
                        "target_semantics": "unresolved",
                        "metadata_evidence": None,
                    }
                )
        except (OSError, ValueError) as error:
            failures.append({"relative_source_path": relative, "error": str(error)})
    report = {"schema_version": 1, "runs": records, "failures": failures}
    write_json(output, report)
    return report


def prepare(config: dict, root: Path) -> dict:
    """One sample per run, with all channels kept together; no automatic averaging."""
    source = (root / config["source"]).resolve()
    review = load_json((root / config["review"]).resolve())
    output = (root / config["prepared"]).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new prepared directory: {output}")
    task = config.get("task", "exploratory")
    if task not in {"exploratory", "primary"}:
        raise ValueError("task must be exploratory or primary")
    count = config.get("positive_frequency_count", 64)
    if type(count) is not int or count < 1:
        raise ValueError("positive_frequency_count must be a positive integer")
    inputs, targets, meshes, records, excluded = [], [], [], [], []
    layout = None
    for reviewed in review["runs"]:
        run = {**config.get("selection", {}), **reviewed}
        # Null fields in a catalog template can be supplied once in YAML.
        for key, value in config.get("selection", {}).items():
            if run.get(key) is None or run.get(key) == "unresolved":
                run[key] = value
        if run.get("include", config.get("include_all", False)) is not True:
            continue
        relative = run["relative_source_path"]
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Invalid source path: {relative}")
        try:
            group = run.get("physical_state_id") or run.get("replica_group_id")
            if (
                not present(group)
                and task == "exploratory"
                and config.get("grouping") == "material_beta_candidate"
            ):
                if not present(run.get("material")) or run.get("beta") is None:
                    raise ValueError(
                        "Candidate grouping requires observed material path and beta"
                    )
                group = f"candidate:{run['material']}:beta={run['beta']}"
                run["grouping_evidence"] = "conservative_path_and_mesh_candidate_only"
            if not present(group):
                raise ValueError(
                    "Assign physical_state_id or a conservative replica_group_id"
                )
            digest = sha256_file(path)
            convergence = run.get("convergence", {})
            verified = (
                convergence.get("status") == "verified_converged"
                and present(convergence.get("evidence_source"))
                and convergence.get("source_sha256") == digest
                and convergence.get("target_path") == run["target_path"]
            )
            if task == "primary":
                if not verified or not present(run.get("physical_state_id")):
                    raise ValueError(
                        "Primary experiment needs a reviewed state and scoped convergence evidence"
                    )
                if not present(run.get("metadata_evidence")) or run.get(
                    "input_semantics"
                ) not in {
                    "dft_derived_g0",
                    "dmft_weiss",
                    "aim_hybridization",
                }:
                    raise ValueError(
                        "Primary experiment needs reviewed input semantics and metadata evidence"
                    )
                if run.get("target_semantics") not in {
                    "impurity_self_energy",
                    "lattice_self_energy",
                }:
                    raise ValueError(
                        "Primary experiment needs reviewed target self-energy semantics"
                    )
            if run.get("input_path") == run.get("target_path"):
                raise ValueError("Input and target paths must differ")
            with h5py.File(path, "r") as handle:
                x = read_quantity(handle, run["input_path"], count)
                y = read_quantity(handle, run["target_path"], count)
            if not np.array_equal(x.mesh, y.mesh):
                raise ValueError("Paired input and target frequency meshes differ")
            current = (x.blocks, y.blocks, x.values.shape, y.values.shape)
            if layout is not None and current != layout:
                raise ValueError(
                    "Channel layout differs; use a separate experiment or explicit basis mapping"
                )
            layout = current
            inputs.append(x.values)
            targets.append(y.values)
            meshes.append(x.mesh)
            target_hash = hashlib.sha256(
                y.values.tobytes() + y.mesh.tobytes()
            ).hexdigest()
            pair_hash = hashlib.sha256(
                x.values.tobytes() + y.values.tobytes() + y.mesh.tobytes()
            ).hexdigest()
            records.append(
                {
                    **run,
                    "group_id": group,
                    "source_sha256": digest,
                    "input_dataset_paths": x.paths,
                    "target_dataset_paths": y.paths,
                    "target_payload_sha256": target_hash,
                    "pair_payload_sha256": pair_hash,
                    "convergence_status": "verified_converged"
                    if verified
                    else "not_assessed",
                }
            )
        except (ValueError, KeyError, OSError) as error:
            excluded.append({"relative_source_path": relative, "reason": str(error)})
    report = {
        "schema_version": 1,
        "task": task,
        "records": records,
        "excluded": excluded,
        "sample_count": len(records),
        "group_count": len({r["group_id"] for r in records}),
        "positive_frequency_count": count,
        "config": config,
        "input_blocks": layout[0] if layout else [],
        "target_blocks": layout[1] if layout else [],
        "interpretation": (
            "external_evidence_reviewed_targets"
            if task == "primary"
            else "stored_array_mapping_with_unverified_physical_targets"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "manifest.json", report)
    if inputs:
        np.savez_compressed(
            output / "arrays.npz",
            inputs=np.stack(inputs),
            targets=np.stack(targets),
            meshes=np.stack(meshes),
        )
        report["arrays_sha256"] = sha256_file(output / "arrays.npz")
        write_json(output / "manifest.json", report)
        replica_summary = []
        for group in sorted({r["group_id"] for r in records}):
            indices = [i for i, r in enumerate(records) if r["group_id"] == group]
            values = np.stack([targets[i] for i in indices])
            replica_summary.append(
                {
                    "group_id": group,
                    "run_count": len(indices),
                    "sources": [records[i]["relative_source_path"] for i in indices],
                    "unique_target_payloads": len(
                        {records[i]["target_payload_sha256"] for i in indices}
                    ),
                    "target_rms_spread": float(
                        np.sqrt(np.mean(np.abs(values - values.mean(axis=0)) ** 2))
                    ),
                    "interpretation": "descriptive_run_spread_not_QMC_uncertainty",
                }
            )
        write_json(output / "replicas.json", replica_summary)
    return report

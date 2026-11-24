"""Conservative admission gate for starting ML experiments.

The gate evaluates already-produced reports.  It does not inspect HDF5 data,
infer physical metadata, or establish DMFT convergence on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"_error": f"missing report: {report_path}"}
    except json.JSONDecodeError as error:
        return {"_error": f"invalid JSON report {report_path}: {error.msg}"}
    if not isinstance(value, dict):
        return {"_error": f"report must be a JSON object: {report_path}"}
    return value


def assess_ml_readiness(dataset_build_report: str | Path) -> dict[str, Any]:
    """Return a fail-closed readiness decision from a dataset-build report.

    A ready decision requires explicitly verified converged records and at
    least one non-random, state-blocked holdout.  This is an engineering gate,
    not a scientific validation result.
    """
    report = _load_report(dataset_build_report)
    blockers: list[str] = []
    if "_error" in report:
        blockers.append(report["_error"])
        return {
            "status": "not_ready",
            "blockers": blockers,
            "limitations": [
                "No conclusion about HDF5 validity or physical convergence was made."
            ],
        }

    if report.get("status") != "passed":
        blockers.append(
            f"dataset-build status is {report.get('status', 'missing')!r}, not 'passed'"
        )
    if report.get("warnings"):
        blockers.append("dataset-build report contains unresolved warnings")

    for label in ("baseline", "physics_constrained"):
        record_count = report.get(label, {}).get("record_count", 0)
        if not isinstance(record_count, int) or record_count < 1:
            blockers.append(f"{label} has no explicitly verified converged records")

    splits = report.get("blocked_splits", {})
    if splits.get("status") != "created":
        blockers.append("physical-state blocked splits were not created")
    material_split = splits.get("splits", {}).get("material", {})
    if material_split.get("status") != "created" or not material_split.get("folds"):
        blockers.append("no material-blocked holdout is available")
    else:
        for fold in material_split["folds"]:
            train = set(fold.get("train_state_ids", []))
            held_out = set(fold.get("held_out_state_ids", []))
            if not train or not held_out or train & held_out:
                blockers.append("a material-blocked fold has empty or overlapping states")
                break

    return {
        "status": "ready" if not blockers else "not_ready",
        "blockers": blockers,
        "evidence": {
            "dataset_build_status": report.get("status"),
            "baseline_records": report.get("baseline", {}).get("record_count", 0),
            "physics_constrained_records": report.get(
                "physics_constrained", {}
            ).get("record_count", 0),
            "independent_physical_state_count": splits.get(
                "independent_physical_state_count", 0
            ),
            "material_blocked_split_status": material_split.get("status"),
        },
        "limitations": [
            "This gate does not independently validate convergence, metadata, "
            "causality, or real-frequency spectra.",
            "Temperature- and parameter-blocked holdouts remain required when "
            "those metadata are available and relevant to the claim.",
        ],
    }


def write_ml_readiness_report(
    dataset_build_report: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Assess readiness and write a compact JSON decision report."""
    decision = assess_ml_readiness(dataset_build_report)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return decision

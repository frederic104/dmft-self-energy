"""Tests for raw-run provenance and fail-closed research readiness audits."""

import json
from pathlib import Path

import h5py

from dmft_data.research_audit import (
    audit_replicas,
    build_physical_state_manifest,
    build_target_registry,
    create_blocked_splits,
)


def _file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_group("dmft_output")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_manifest_preserves_unresolved_state_and_storage_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data" / "raw" / "hdf5"
    _file(root / "BVO" / "run-a.h5")
    _file(root / "BVO" / "run-b.h5")
    diagnostics = tmp_path / "diagnostics.json"
    _write_json(diagnostics, [
        {
            "relative_source_path": "BVO/run-a.h5",
            "material": "BVO",
            "nwarmup": 100000,
            "mesh_metadata": {"beta": 40.0},
            "run_status": "final_matches_earlier_iteration",
        },
        {
            "relative_source_path": "BVO/run-b.h5",
            "material": "BVO",
            "nwarmup": 150000,
            "mesh_metadata": {"beta": 40.0},
            "run_status": "final_matches_earlier_iteration",
        },
    ])
    metadata = tmp_path / "states.json"
    _write_json(metadata, {
        "runs": {
            "BVO/run-a.h5": {"physical_state_id": "bvo-beta40"},
            "BVO/run-b.h5": {"physical_state_id": "bvo-beta40"},
        },
        "physical_states": {"bvo-beta40": {"temperature": 290}},
    })

    manifest = build_physical_state_manifest(
        root, tmp_path / "manifest.json", diagnostics_path=diagnostics,
        metadata_path=metadata,
    )
    audit = audit_replicas(tmp_path / "manifest.json", tmp_path / "replicas.json")

    assert manifest["record_count"] == 2
    assert manifest["records"][0]["convergence_status"] == "not_assessed"
    assert manifest["records"][0]["storage_status"] == "final_matches_earlier_iteration"
    assert manifest["records"][0]["hdf5_storage_schema"]["status"] == "observed"
    assert manifest["records"][0]["hdf5_storage_schema"]["final_blocks"] == {
        "G_iw": [],
        "Sigma_iw": [],
    }
    assert audit["replica_groups"][0]["member_count"] == 2
    assert audit["replica_groups"][0]["nwarmup_values"] == ["100000", "150000"]


def test_target_registry_and_splits_require_explicit_external_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {
        "records": [
            {
                "relative_source_path": "SVO/a.h5",
                "physical_state_id": "svo-state",
                "source_sha256": "a" * 64,
                "material": "SVO",
                "storage_status": "final_matches_last_iteration",
            },
            {
                "relative_source_path": "CVO/b.h5",
                "physical_state_id": "cvo-state",
                "source_sha256": "b" * 64,
                "material": "CVO",
                "storage_status": "final_matches_last_iteration",
            },
        ]
    })

    no_evidence = build_target_registry(manifest_path, tmp_path / "targets.json")
    no_split = create_blocked_splits(
        manifest_path, tmp_path / "targets.json", tmp_path / "splits.json",
        minimum_states=2,
    )
    assert no_evidence["primary_target_count"] == 0
    assert no_split["status"] == "insufficient_verified_physical_states"

    registry = tmp_path / "convergence.json"
    _write_json(registry, {"runs": {
        "SVO/a.h5": {
            "status": "verified_converged",
            "evidence_source": "solver-log: line 42",
            "selected_iteration": 15,
            "source_sha256": "a" * 64,
            "selected_quantity_paths": {"Sigma_iw": "/dmft_output/Sigma-15"},
        },
        "CVO/b.h5": {
            "status": "verified_converged",
            "evidence_source": "solver-log: line 51",
            "selected_iteration": 15,
            "source_sha256": "b" * 64,
            "selected_quantity_paths": {"Sigma_iw": "/dmft_output/Sigma-15"},
        },
    }})
    targets = build_target_registry(
        manifest_path, tmp_path / "verified_targets.json",
        convergence_registry_path=registry,
    )
    splits = create_blocked_splits(
        manifest_path, tmp_path / "verified_targets.json",
        tmp_path / "verified_splits.json", minimum_states=2,
    )
    assert targets["primary_target_count"] == 2
    assert splits["status"] == "created"
    assert {fold["held_out_material"] for fold in splits["folds"]} == {"SVO", "CVO"}

"""Tests for the fail-closed ML readiness gate."""

import json
from pathlib import Path

from dmft_data.ml_readiness import assess_ml_readiness, write_ml_readiness_report


def _report() -> dict:
    return {
        "status": "passed",
        "warnings": [],
        "baseline": {"record_count": 3},
        "physics_constrained": {"record_count": 6},
        "blocked_splits": {
            "status": "created",
            "independent_physical_state_count": 3,
            "splits": {
                "material": {
                    "status": "created",
                    "folds": [{
                        "train_state_ids": ["state-A", "state-B"],
                        "held_out_state_ids": ["state-C"],
                    }],
                }
            },
        },
    }


def test_ready_requires_non_overlapping_material_holdout(tmp_path: Path) -> None:
    input_path = tmp_path / "research_dataset_build.json"
    input_path.write_text(json.dumps(_report()), encoding="utf-8")

    decision = assess_ml_readiness(input_path)

    assert decision["status"] == "ready"
    assert decision["blockers"] == []


def test_missing_or_limited_report_fails_closed(tmp_path: Path) -> None:
    output_path = tmp_path / "ml_readiness.json"
    decision = write_ml_readiness_report(tmp_path / "missing.json", output_path)

    assert decision["status"] == "not_ready"
    assert "missing report" in decision["blockers"][0]
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "not_ready"


def test_overlapping_holdout_is_blocked(tmp_path: Path) -> None:
    report = _report()
    report["blocked_splits"]["splits"]["material"]["folds"][0][
        "held_out_state_ids"
    ] = ["state-A"]
    input_path = tmp_path / "research_dataset_build.json"
    input_path.write_text(json.dumps(report), encoding="utf-8")

    decision = assess_ml_readiness(input_path)

    assert decision["status"] == "not_ready"
    assert "overlapping" in decision["blockers"][-1]

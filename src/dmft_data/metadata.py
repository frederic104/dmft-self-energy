"""Normalization of run and physical-state identifiers."""

from __future__ import annotations

from typing import Any

UNKNOWN = "unknown"


def _explicit_attribute(attributes: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    normalized = {str(key).lower(): value for key, value in attributes.items()}
    for key in keys:
        if key in normalized and normalized[key] not in (None, ""):
            return str(normalized[key])
    return None


def normalize_run_and_state(
    file_attributes: dict[str, Any],
    relative_source_path: str,
) -> tuple[str, str]:
    """Return explicit identifiers; infer no physical state or run semantics."""
    run_id = _explicit_attribute(
        file_attributes,
        ("run_id", "calculation_run_id", "calculation_id", "run"),
    )
    if run_id is None:
        run_id = relative_source_path

    physical_state_id = _explicit_attribute(
        file_attributes,
        ("physical_state_id", "state_id", "calculation_state_id"),
    )
    if physical_state_id is None:
        physical_state_id = UNKNOWN
    return run_id, physical_state_id

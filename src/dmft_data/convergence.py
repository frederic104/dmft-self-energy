"""Physical convergence assessment.

This module intentionally has no default convergence criterion.
"""

from __future__ import annotations

from typing import Any


def assess_convergence(validation_config: dict[str, Any]) -> dict[str, Any]:
    """Record convergence status without making a physical convergence claim."""
    convergence = validation_config.get("convergence", {})
    if not isinstance(convergence, dict):
        convergence = {}
    if convergence.get("enabled") is not True or convergence.get("criterion") is None:
        return {
            "status": "not_assessed",
            "criterion": convergence.get("criterion"),
            "threshold": convergence.get("threshold"),
            "last_iteration_is_converged": False,
            "reason": "No explicit verified convergence criterion was supplied.",
        }
    return {
        "status": "not_assessed",
        "criterion": convergence.get("criterion"),
        "threshold": convergence.get("threshold"),
        "last_iteration_is_converged": False,
        "reason": "Automated convergence assessment is not enabled.",
    }

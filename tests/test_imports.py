"""Basic import checks for the initial package structure."""

import importlib


def test_placeholder_modules_import() -> None:
    """All initial dmft_data modules should be importable."""
    module_names = (
        "convergence",
        "dataset",
        "extraction",
        "inventory",
        "metadata",
        "provenance",
        "validation",
    )

    for module_name in module_names:
        importlib.import_module(f"dmft_data.{module_name}")

"""Provenance helpers for data processing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def redact_source_root(value: Any, source_root: str | Path) -> Any:
    """Hide the supplied filesystem root in a report, retaining HDF5 paths.

    This is scoped redaction, not an assertion that arbitrary attributes contain
    no private information. Raw source attributes/files are never modified.
    """
    root = str(source_root)
    variants = {root, root.replace("\\", "/"), root.replace("/", "\\")}
    if isinstance(value, str):
        for prefix in sorted(variants, key=len, reverse=True):
            value = value.replace(prefix, "<LOCAL_SOURCE_ROOT>")
        return value
    if isinstance(value, list):
        return [redact_source_root(v, source_root) for v in value]
    if isinstance(value, dict):
        return {redact_source_root(k, source_root): redact_source_root(v, source_root)
                for k, v in value.items()}
    return value


def sha256_file(path: str | Path) -> str:
    """Return a file SHA-256 checksum."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    """Return extraction dependency versions."""
    return {"h5py": h5py.__version__, "numpy": np.__version__}

"""Provenance helpers for data processing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np


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

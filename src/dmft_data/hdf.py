"""Shared, read-only access to explicitly selected TRIQS BlockGf quantities."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from .extraction import (
    UnsupportedSchemaError,
    _associate_mesh,
    _attributes,
    _find_meshes,
    _reconstruct_complex,
)


def read_complex_dataset(dataset: h5py.Dataset) -> np.ndarray:
    """Use the extractor's single implementation of complex reconstruction."""
    values, _ = _reconstruct_complex(dataset.file, dataset)
    if values.size == 0 or not np.isfinite(values).all():
        raise UnsupportedSchemaError(f"{dataset.name}: empty or non-finite payload")
    return np.asarray(values, dtype=np.complex128)


def physical_blocks(group: h5py.Group) -> dict[str, h5py.Dataset]:
    return {
        name: block["data"]
        for name, block in sorted(group.items())
        if isinstance(block, h5py.Group) and isinstance(block.get("data"), h5py.Dataset)
    }


@dataclass
class Quantity:
    values: np.ndarray  # block, matrix row, matrix column, frequency
    mesh: np.ndarray
    blocks: list[str]
    paths: list[str]


def read_quantity(handle: h5py.File, path: str, count: int) -> Quantity:
    """Read local block meshes and the first count positive frequencies.

    Selection preserves stored ordering, requires increasing positive imaginary
    frequencies, and performs no interpolation. All blocks must share a mesh.
    """
    root = handle.get(path)
    if not isinstance(root, h5py.Group):
        raise UnsupportedSchemaError(f"Quantity group unavailable: {path}")
    blocks = physical_blocks(root)
    if not blocks:
        raise UnsupportedSchemaError(f"No physical data blocks: {path}")
    arrays, paths = [], []
    common_mesh = None
    for name, dataset in blocks.items():
        values = read_complex_dataset(dataset)
        meshes = _find_meshes(dataset.parent)
        mesh, axis = _associate_mesh(dataset.name, values, _attributes(dataset), meshes)
        if mesh is None or axis is None:
            raise UnsupportedSchemaError(
                f"Local Matsubara mesh missing: {dataset.name}"
            )
        if not np.allclose(mesh.values.real, 0, atol=1e-14, rtol=0):
            raise UnsupportedSchemaError(
                f"Mesh must be imaginary Matsubara: {mesh.source_path}"
            )
        positive = np.flatnonzero(mesh.values.imag > 0)
        selected = mesh.values[positive[:count]]
        if len(selected) != count or np.any(np.diff(selected.imag) <= 0):
            raise UnsupportedSchemaError(
                f"Need {count} ordered positive frequencies: {path}/{name}"
            )
        if common_mesh is not None and not np.array_equal(common_mesh, selected):
            raise UnsupportedSchemaError(f"Block meshes differ: {path}")
        common_mesh = selected
        values = np.take(np.moveaxis(values, axis, -1), positive[:count], axis=-1)
        if values.ndim == 1:
            values = values[None, None, :]
        if values.ndim != 3 or values.shape[0] != values.shape[1]:
            raise UnsupportedSchemaError(
                f"Expected square block matrices: {dataset.name}"
            )
        arrays.append(values)
        paths.append(dataset.name)
    if len({x.shape for x in arrays}) != 1:
        raise UnsupportedSchemaError(
            f"Mixed matrix sizes need separate experiments: {path}"
        )
    return Quantity(np.stack(arrays), common_mesh, list(blocks), paths)

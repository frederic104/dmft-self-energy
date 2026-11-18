"""Tests for the representative TRIQS/DFTTools HDF5 reader."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from dmft_data.extraction import (
    ComplexRepresentationError,
    UnsupportedSchemaError,
    read_hdf5_file,
)


def test_extracts_native_complex_arrays_mesh_iterations_and_channels(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    path = source_root / "synthetic.h5"
    iw = np.array([1j * (2 * n + 1) * np.pi * 0.1 for n in range(3)])

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "synthetic-triqs"
        dmft = handle.create_group("dmft")
        dmft.create_dataset("iw", data=iw)
        sigma = dmft.create_dataset(
            "iteration_0007/Sigma_imp",
            data=np.ones((2, 2, 3), dtype=np.complex128),
        )
        sigma.attrs["spin_labels"] = ["up", "down"]
        sigma.attrs["orbital_labels"] = ["dxz", "dyz"]
        sigma.attrs["frequency_axis"] = 2
        dmft.create_dataset(
            "iteration_0007/G0",
            data=np.zeros((2, 2, 3), dtype=np.complex128),
        )
        dmft.create_dataset(
            "iteration_0007/Weiss",
            data=np.zeros((2, 2, 3), dtype=np.complex128),
        )
        dmft.create_dataset(
            "iteration_0007/hybrid",
            data=np.zeros((2, 2, 3), dtype=np.complex128),
        )

    result = read_hdf5_file(path, source_root)

    assert result.relative_source_path == "synthetic.h5"
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.file_attributes["format"] == "synthetic-triqs"
    assert len(result.meshes) == 1
    assert np.allclose(result.meshes[0].values, iw)
    assert len(result.arrays) == 4

    by_role = {array.role: array for array in result.arrays}
    assert by_role["dft_derived_g0"].source_path == "/dmft/iteration_0007/G0"
    assert by_role["dmft_weiss"].source_path == "/dmft/iteration_0007/Weiss"
    assert by_role["aim_hybridization"].source_path == "/dmft/iteration_0007/hybrid"
    sigma = by_role["impurity_self_energy"]
    assert sigma.iteration == "0007"
    assert sigma.frequency_axis == 2
    assert sigma.spin_labels == ["up", "down"]
    assert sigma.orbital_labels == ["dxz", "dyz"]
    assert np.isfinite(sigma.values).all()


def test_compound_real_imaginary_representation_is_reconstructed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compound.h5"
    iw = np.array([-3.0, -1.0, 1.0, 3.0])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("dmft")
        group.create_dataset("iw", data=iw)
        compound = np.zeros((2, 2, 4), dtype=[("r", "f8"), ("i", "f8")])
        compound["r"] = 1.0
        compound["i"] = 2.0
        group.create_dataset("iteration_3/Sigma", data=compound)

    result = read_hdf5_file(path)
    sigma = result.arrays[0]

    assert sigma.values.shape == (2, 2, 4)
    assert np.allclose(sigma.values.real, 1.0)
    assert np.allclose(sigma.values.imag, 2.0)
    assert "complex" in sigma.stored_dtype or "compound" in sigma.stored_dtype
    assert sigma.iteration == "3"


def test_sibling_real_and_imaginary_datasets_are_reconstructed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mesh", data=np.arange(4, dtype=np.float64))
        handle.create_dataset("G_real", data=np.ones((2, 4)))
        handle.create_dataset("G_imag", data=np.full((2, 4), 2.0))

    result = read_hdf5_file(path)

    assert len(result.arrays) == 1
    real = next(array for array in result.arrays if array.source_path == "/G_real")
    assert np.allclose(real.values.real, 1.0)
    assert np.allclose(real.values.imag, 2.0)


def test_inline_mesh_attribute_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "inline.h5"
    iw = np.array([1j, 3j, 5j])
    with h5py.File(path, "w") as handle:
        g = handle.create_dataset(
            "dmft/G",
            data=np.zeros((2, 2, 3), dtype=np.complex128),
        )
        g.attrs["matsubara_frequencies"] = iw

    result = read_hdf5_file(path)
    g = result.arrays[0]

    assert g.frequency_mesh is not None
    assert g.frequency_mesh.source_path == "/dmft/G"
    assert g.frequency_mesh.attributes["mesh_attribute"] == "matsubara_frequencies"
    assert np.allclose(g.frequency_mesh.values, iw)


def test_shape_mismatch_raises_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_shape.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("omega", data=np.arange(5, dtype=np.float64))
        handle.create_dataset(
            "dmft/Sigma",
            data=np.zeros((2, 2, 3), dtype=np.complex128),
            )

    with pytest.raises(UnsupportedSchemaError, match="no unique Matsubara mesh axis"):
        read_hdf5_file(path)


def test_unknown_compound_dtype_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unknown_compound.h5"
    compound = np.zeros((3,), dtype=[("x", "f8"), ("y", "f8")])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iw", data=np.arange(3, dtype=np.float64))
        handle.create_dataset("G", data=compound)

    with pytest.raises(ComplexRepresentationError, match="compound dtype fields"):
        read_hdf5_file(path)

"""Tests for the representative TRIQS/DFTTools HDF5 reader."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from dmft_data.extraction import (
    ComplexRepresentationError,
    UnsupportedSchemaError,
    read_hdf5_file,
    run_extraction,
    write_normalized_hdf5,
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


def test_scalar_beta_is_ignored_when_one_dimensional_mesh_is_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "domain_beta.h5"
    iw = np.array([1j * (2 * n + 1) * np.pi for n in range(3)])
    with h5py.File(path, "w") as handle:
        beta_group = handle.create_group(
            "dmft_output/G-1/down_0/mesh/domain"
        )
        beta_group.create_dataset("beta", data=np.float64(40.0))
        handle.create_dataset(
            "dmft_output/G-1/down_0/mesh/MeshImFreq",
            data=iw,
        )
        handle.create_dataset(
            "dmft_output/G-1/down_0/G",
            data=np.ones(3, dtype=np.complex128),
        )

    result = read_hdf5_file(path)

    assert len(result.meshes) == 1
    assert result.meshes[0].source_path == (
        "/dmft_output/G-1/down_0/mesh/MeshImFreq"
    )
    assert np.allclose(result.meshes[0].values, iw)
    assert len(result.arrays) == 1
    candidate = result.arrays[0]
    assert candidate.source_path == "/dmft_output/G-1/down_0/G"
    assert candidate.frequency_mesh is not None
    assert candidate.frequency_mesh.source_path == result.meshes[0].source_path
    assert candidate.frequency_axis == 0


def test_object_metadata_matching_g0_path_is_not_a_candidate_array(
    tmp_path: Path,
) -> None:
    path = tmp_path / "g0_block_names.h5"
    iw = np.array([1j * (2 * n + 1) * np.pi for n in range(3)])
    with h5py.File(path, "w") as handle:
        mesh = handle.create_group("dmft_output/G0-1/mesh")
        mesh.create_dataset("MeshImFreq", data=iw)
        g0_metadata = handle.create_group("dmft_output/G0-1/block_names")
        g0_metadata.create_dataset("0", data="up", dtype=h5py.string_dtype())
        handle.create_dataset(
            "dmft_output/G0-1/G0",
            data=np.zeros(3, dtype=np.complex128),
        )

    result = read_hdf5_file(path)

    assert len(result.meshes) == 1
    assert len(result.arrays) == 1
    candidate = result.arrays[0]
    assert candidate.source_path == "/dmft_output/G0-1/G0"
    assert candidate.role == "dft_derived_g0"
    assert candidate.frequency_mesh is not None
    assert candidate.frequency_mesh.source_path == (
        "/dmft_output/G0-1/mesh/MeshImFreq"
    )
    assert candidate.frequency_axis == 0


def test_triqs_blockgf_last_axis_complex_without_mesh_vector(
    tmp_path: Path,
) -> None:
    path = tmp_path / "triqs_blockgf.h5"
    raw = np.zeros((2050, 1, 1, 2), dtype=np.float64)
    raw[..., 0] = 1.0
    raw[..., 1] = 2.0
    with h5py.File(path, "w") as handle:
        blockgf = handle.create_group("dmft_output/G_iw")
        block = blockgf.create_group("down_0")
        data = block.create_dataset("data", data=raw)
        data.attrs["__complex__"] = 1
        mesh = block.create_group("mesh")
        domain = mesh.create_group("domain")
        domain.create_dataset("beta", data=np.float64(40.0))
        domain.create_dataset(
            "statistic",
            data="Fermionic",
            dtype=h5py.string_dtype(),
        )
        mesh.create_dataset("size", data=np.int64(2050))
        mesh.create_dataset("positive_freq_only", data=np.int32(0))

    result = read_hdf5_file(path)

    assert len(result.meshes) == 1
    mesh = result.meshes[0]
    assert mesh.source_path == "/dmft_output/G_iw/down_0/mesh"
    assert mesh.attributes["mesh_format"] == "TRIQS MeshImFreq"
    assert mesh.attributes["reconstructed"] is True
    assert mesh.values.shape == (2050,)
    assert np.isclose(mesh.values[0].imag, -(2049.0 * np.pi / 40.0))
    assert np.isclose(mesh.values[-1].imag, (2049.0 * np.pi / 40.0))
    assert np.isclose(mesh.values[1025].imag, np.pi / 40.0)
    assert len(result.arrays) == 1
    candidate = result.arrays[0]
    assert candidate.source_path == "/dmft_output/G_iw/down_0/data"
    assert candidate.values.shape == (2050, 1, 1)
    assert np.iscomplexobj(candidate.values)
    assert np.all(candidate.values.real == 1.0)
    assert np.all(candidate.values.imag == 2.0)
    assert candidate.frequency_mesh is mesh
    assert candidate.frequency_axis == 0


def test_triqs_g_tau_is_not_assigned_matsubara_mesh(
    tmp_path: Path,
) -> None:
    path = tmp_path / "triqs_g_tau.h5"
    iw_data = np.zeros((2050, 1, 1, 2), dtype=np.float64)
    tau_data = np.zeros((10001, 1, 1, 2), dtype=np.float64)
    iw_data[..., 0] = 1.0
    iw_data[..., 1] = 2.0
    tau_data[..., 0] = 3.0
    tau_data[..., 1] = 4.0
    with h5py.File(path, "w") as handle:
        dmft = handle.create_group("dmft_output")
        iw_block = dmft.create_group("G_iw/down_0")
        iw_block.create_dataset("data", data=iw_data)
        iw_block["data"].attrs["__complex__"] = 1
        mesh = dmft.create_group("G_iw/down_0/mesh")
        domain = mesh.create_group("domain")
        domain.create_dataset("beta", data=np.float64(40.0))
        domain.create_dataset("statistic", data="F", dtype=h5py.string_dtype())
        mesh.create_dataset("size", data=np.int64(2050))
        mesh.create_dataset("positive_freq_only", data=np.int32(0))

        tau_block = dmft.create_group("G_tau/down_0")
        tau_block.create_dataset("data", data=tau_data)
        tau_block["data"].attrs["__complex__"] = 1

    result = read_hdf5_file(path)

    assert len(result.arrays) == 1
    candidate = result.arrays[0]
    assert candidate.source_path == "/dmft_output/G_iw/down_0/data"
    assert candidate.frequency_mesh is not None
    assert candidate.frequency_mesh.values.shape == (2050,)
    assert candidate.frequency_axis == 0


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


def test_normalized_output_is_idempotent_and_separates_identifiers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "SVO" / "run.h5"
    source.parent.mkdir(parents=True)
    iw = np.array([1j, 3j, 5j])
    with h5py.File(source, "w") as handle:
        handle.attrs["run_id"] = "run-17"
        handle.attrs["physical_state_id"] = "state-A"
        handle.create_dataset("iw", data=iw)
        handle.create_dataset(
            "dmft/Sigma",
            data=np.ones((2, 3), dtype=np.complex128),
        )

    extraction = read_hdf5_file(source, tmp_path / "raw")
    output = tmp_path / "processed" / "arrays" / "SVO" / "run.h5"
    first_path, first = write_normalized_hdf5(
        extraction,
        output,
        {"preserve_complex_values": True},
    )
    second_path, second = write_normalized_hdf5(
        extraction,
        output,
        {"preserve_complex_values": True},
    )

    assert first_path == second_path == output
    assert first["status"] == "written"
    assert second["status"] == "unchanged"
    assert second["rewritten"] is False
    with h5py.File(output, "r") as handle:
        assert handle.attrs["run_id"] == "run-17"
        assert handle.attrs["physical_state_id"] == "state-A"
        assert handle["run/provenance"].attrs["sha256"] == extraction.sha256
        assert np.iscomplexobj(handle["run/arrays/array_0000"][()])

    with pytest.raises(FileExistsError, match="Refusing silent overwrite"):
        write_normalized_hdf5(extraction, output, {"different": True})


def test_normalized_output_derives_run_and_verified_mesh_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "SVO" / "SVO_beta40.h5"
    source.parent.mkdir(parents=True)
    iw = np.array([1j * (2 * n + 1) * np.pi / 40.0 for n in range(2050)])
    with h5py.File(source, "w") as handle:
        dmft = handle.create_group("dmft_output/G_iw/down_0")
        dmft.create_dataset("data", data=np.zeros((2050, 1, 1, 2)))
        dmft["data"].attrs["__complex__"] = 1
        mesh = dmft.create_group("mesh")
        domain = mesh.create_group("domain")
        domain.create_dataset("beta", data=np.float64(40.0))
        domain.create_dataset("statistic", data="F", dtype=h5py.string_dtype())
        mesh.create_dataset("size", data=np.int64(2050))
        mesh.create_dataset("positive_freq_only", data=np.int32(0))

    extraction = read_hdf5_file(source, tmp_path / "raw")
    output = tmp_path / "processed" / "arrays" / "SVO" / "SVO_beta40.h5"
    _, result = write_normalized_hdf5(extraction, output)

    assert result["run_id"] == "SVO/SVO_beta40.h5"
    assert result["physical_state_id"] == "material=SVO,beta=40"
    with h5py.File(output, "r") as handle:
        assert bool(handle.attrs["run_id_explicit"]) is True
        assert bool(handle.attrs["physical_state_id_explicit"]) is True
        assert handle.attrs["material"] == "SVO"
        assert handle.attrs["beta"] == 40.0
        assert handle.attrs["parameter_block"] == "beta=40"


def test_normalized_output_updates_metadata_for_same_source_and_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "SVO" / "run.h5"
    source.parent.mkdir(parents=True)
    iw = np.array([1j, 3j, 5j])
    with h5py.File(source, "w") as handle:
        dmft = handle.create_group("dmft_output/G_iw/down_0")
        dmft.create_dataset("data", data=np.zeros((2050, 1, 1, 2)))
        dmft["data"].attrs["__complex__"] = 1
        mesh = dmft.create_group("mesh")
        domain = mesh.create_group("domain")
        domain.create_dataset("beta", data=np.float64(40.0))
        domain.create_dataset("statistic", data="F", dtype=h5py.string_dtype())
        mesh.create_dataset("size", data=np.int64(2050))
        mesh.create_dataset("positive_freq_only", data=np.int32(0))

    extraction = read_hdf5_file(source, tmp_path / "raw")
    output = tmp_path / "processed" / "arrays" / "SVO" / "run.h5"
    extraction_config = {"preserve_complex_values": True}
    _, first = write_normalized_hdf5(
        extraction,
        output,
        extraction_config,
    )
    assert first["status"] == "written"

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    old_manifest = __import__("json").loads(
        manifest_path.read_text(encoding="utf-8")
    )
    old_manifest["physical_state_id"] = "unknown"
    manifest_path.write_text(
        __import__("json").dumps(old_manifest),
        encoding="utf-8",
    )

    _, updated = write_normalized_hdf5(
        extraction,
        output,
        extraction_config,
    )

    assert updated["status"] == "written"
    assert updated["rewritten"] is True
    assert updated["physical_state_id"] == "material=SVO,beta=40"


def test_failed_extraction_is_reported_without_silently_skipping(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    raw = root / "data" / "raw" / "hdf5"
    raw.mkdir(parents=True)
    (raw / "broken.h5").write_text("not hdf5", encoding="utf-8")
    config = root / "configs" / "dataset.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump({
            "paths": {
                "raw_dir": "data/raw/hdf5",
                "processed_dir": "data/processed",
                "report_dir": "reports",
            },
            "dataset": {
                "file_extensions": [".h5", ".hdf5"],
                "recursive": True,
            },
            "extraction": {"preserve_complex_values": True},
        }),
        encoding="utf-8",
    )

    report = run_extraction(config, root)

    assert report["status"] == "failed"
    assert report["files"] == []
    assert report["failures"][0]["error_type"] == "UnsupportedSchemaError"
    assert (root / "reports" / "hdf5_extraction.json").is_file()

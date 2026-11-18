"""Representative TRIQS/DFTTools HDF5 extraction.

This module extracts only representations that can be verified from the HDF5
schema.  It deliberately does not infer convergence, chemical parameters,
solver metadata, or physical channel meanings beyond explicit path or attribute
evidence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class UnsupportedSchemaError(ValueError):
    """Raised when an HDF5 dataset representation is not supported."""


class ComplexRepresentationError(UnsupportedSchemaError):
    """Raised when a stored complex representation is ambiguous or invalid."""


def _json_value(value: Any) -> Any:
    """Convert an HDF5 attribute value to a JSON-compatible value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attributes(handle: h5py.AttributeManager | h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Serialize an object's attributes without changing their source values."""
    attrs: dict[str, Any] = {}
    for key, value in handle.attrs.items():
        attrs[key] = _json_value(value)
    return attrs


def _path_is_mesh(path: str) -> bool:
    name = path.lower()
    terms = ("matsubara", "frequency", "frequencies", "omega", "mesh", "/iw")
    return any(term in name for term in terms)


def _classify_role(path: str, attrs: dict[str, Any]) -> str:
    """Classify a name only when the path provides an explicit role hint."""
    name = path.lower()
    role_attr = str(
        attrs.get("quantity")
        or attrs.get("type")
        or attrs.get("physical_quantity")
        or ""
    ).lower()
    text = f"{name} {role_attr}"

    if "/hybrid" in text or "hybridization" in text:
        return "aim_hybridization"
    if "/weiss" in text or "weiss" in text:
        return "dmft_weiss"
    if "/g0imp" in text or "/g0_imp" in text or "/dft/g0" in text:
        return "dft_derived_g0"
    if re.search(r"/g0(?:$|[_/\s])", text):
        return "dft_derived_g0"
    if "/gimp" in text or "/g_imp" in text or "impurity_green" in text:
        return "impurity_green"
    if "/glat" in text or "/g_lat" in text or "lattice_green" in text:
        return "lattice_green"
    if re.search(r"/g(?:$|[_/])", text):
        return "green_function"
    if "/sigma_imp" in text or "/sigmaimp" in text or "impurity_self_energy" in text:
        return "impurity_self_energy"
    if "/sigma_lat" in text or "/sigma_lat" in text or "lattice_self_energy" in text:
        return "lattice_self_energy"
    if "sigma" in text or "self_energy" in text or "selfenergy" in text:
        return "self_energy"
    return "unknown"


def _is_candidate(path: str) -> bool:
    name = path.lower()
    if _path_is_mesh(name):
        return False
    return (
        re.search(r"/g(?:$|[_./])", name) is not None
        or "/g0" in name
        or name.rstrip("/").endswith("/g")
        or "/gimp" in name
        or "/g_imp" in name
        or "/glat" in name
        or "/g_lat" in name
        or "/weiss" in name
        or "/hybrid" in name
        or "sigma" in name
        or "self_energy" in name
        or "selfenergy" in name
    )


def _labels(attrs: dict[str, Any], kind: str) -> list[str] | None:
    for key in (
        f"{kind}_labels",
        f"{kind}_channels",
        f"{kind}_orbital_names"
        if kind == "orbital"
        else f"{kind}_names",
    ):
        if key in attrs:
            value = attrs[key]
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]
    return None


def _iteration_id(path: str, attrs: dict[str, Any]) -> str | None:
    for key in ("iteration", "iteration_id", "dmft_iteration", "iteration_index"):
        if key in attrs:
            value = attrs[key]
            return str(_json_value(value))
    match = re.search(r"(?:iteration[_-]?|iter[_-]?)(\d+)", path.lower())
    return match.group(1) if match else None


@dataclass(frozen=True)
class FrequencyMesh:
    """An original Matsubara frequency mesh and its source metadata."""

    source_path: str
    values: np.ndarray
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "values": self.values.tolist(),
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class ExtractedArray:
    """One extracted Green function or self-energy array."""

    role: str
    source_path: str
    attributes: dict[str, Any]
    values: np.ndarray
    stored_dtype: str
    frequency_mesh: FrequencyMesh | None
    frequency_axis: int | None
    iteration: str | None
    spin_labels: list[str] | None
    orbital_labels: list[str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_path": self.source_path,
            "attributes": self.attributes,
            "values": self.values.tolist(),
            "stored_dtype": self.stored_dtype,
            "frequency_mesh": (
                self.frequency_mesh.to_dict()
                if self.frequency_mesh is not None
                else None
            ),
            "frequency_axis": self.frequency_axis,
            "iteration": self.iteration,
            "spin_labels": self.spin_labels,
            "orbital_labels": self.orbital_labels,
        }


@dataclass(frozen=True)
class HDF5Extraction:
    """The extracted content of one HDF5 source file."""

    source_path: str
    relative_source_path: str
    sha256: str
    file_attributes: dict[str, Any]
    arrays: tuple[ExtractedArray, ...]
    meshes: tuple[FrequencyMesh, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "relative_source_path": self.relative_source_path,
            "sha256": self.sha256,
            "file_attributes": self.file_attributes,
            "arrays": [array.to_dict() for array in self.arrays],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
        }


def _compound_complex(dataset: h5py.Dataset) -> tuple[np.ndarray, str]:
    dtype = dataset.dtype
    names = tuple(dtype.names or ())
    normalized = {re.sub(r"[^a-z]", "", str(name).lower()): name for name in names}
    real_key = next((normalized[key] for key in normalized if key in {"r", "real"}), None)
    imag_key = next((normalized[key] for key in normalized if key in {"i", "imag", "imaginary"}), None)
    if real_key is None or imag_key is None:
        raise ComplexRepresentationError(
            f"{dataset.name}: compound dtype fields {names} do not verify as real/imaginary"
        )
    raw = dataset[()]
    real = np.asarray(raw[real_key])
    imag = np.asarray(raw[imag_key])
    if real.shape != imag.shape:
        raise ComplexRepresentationError(
            f"{dataset.name}: compound real and imaginary shapes differ"
        )
    if real.dtype.kind not in "iuf" or imag.dtype.kind not in "iuf":
        raise ComplexRepresentationError(
            f"{dataset.name}: real and imaginary fields must be numeric"
        )
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ComplexRepresentationError(
            f"{dataset.name}: reconstructed complex values contain non-finite numbers"
        )
    return real + 1j * imag, f"compound real/imaginary: {names}"


def _reconstruct_complex(
    handle: h5py.File,
    dataset: h5py.Dataset,
) -> tuple[np.ndarray, str]:
    """Verify and reconstruct a stored complex representation."""
    if dataset.dtype.names:
        return _compound_complex(dataset)

    if dataset.dtype.kind == "c":
        values = np.asarray(dataset[()], dtype=np.complex128)
        if not np.all(np.isfinite(values.real)) or not np.all(
            np.isfinite(values.imag)
        ):
            raise ComplexRepresentationError(
                f"{dataset.name}: stored complex values contain non-finite numbers"
            )
        return values, f"native complex: {dataset.dtype}"

    if dataset.dtype.kind not in "iuf":
        raise UnsupportedSchemaError(
            f"{dataset.name}: unsupported dtype {dataset.dtype}"
        )

    name_lower = dataset.name.lower()
    name = dataset.name
    pair: h5py.Dataset | None = None
    base = name
    real_suffixes = ("_real", "/real", ".real", "_r", "/r")
    imag_suffixes = ("_imag", "_imaginary", "/imag", "/imaginary", ".imag", "_i", "/i")
    if any(name_lower.endswith(suffix) for suffix in real_suffixes):
        suffix = next(
            suffix for suffix in real_suffixes if name_lower.endswith(suffix)
        )
        base = name[: -len(suffix)]
        candidates = [base + tail for tail in imag_suffixes]
        pair = next(
            (
                handle[path]
                for path in candidates
                if path in handle and isinstance(handle[path], h5py.Dataset)
            ),
            None,
        )
    if pair is not None:
        real = dataset[()] if name_lower.endswith(tuple(real_suffixes)) else pair[()]
        imag = pair[()] if name_lower.endswith(tuple(real_suffixes)) else dataset[()]
        if np.asarray(real).shape != np.asarray(imag).shape:
            raise ComplexRepresentationError(
                f"{dataset.name}: split real and imaginary shapes differ"
            )
        real_array = np.asarray(real)
        imag_array = np.asarray(imag)
        if real_array.dtype.kind not in "iuf" or imag_array.dtype.kind not in "iuf":
            raise ComplexRepresentationError(
                f"{dataset.name}: split components must be numeric"
            )
        values = real_array + 1j * imag_array
        if not np.all(np.isfinite(values.real)) or not np.all(
            np.isfinite(values.imag)
        ):
            raise ComplexRepresentationError(
                f"{dataset.name}: reconstructed values contain non-finite numbers"
            )
        return values, "sibling real/imaginary datasets"

    if name.endswith(tuple(imag_suffixes)):
        return np.asarray(dataset[()], dtype=np.float64), "real-only input"

    return np.asarray(dataset[()], dtype=np.float64), f"real: {dataset.dtype}"


def _mesh_from_attributes(
    dataset: h5py.Dataset,
    attrs: dict[str, Any],
) -> FrequencyMesh | None:
    for key in ("matsubara_mesh", "matsubara_frequencies", "frequencies", "omega", "mesh"):
        if key not in attrs:
            continue
        value = np.asarray(attrs[key])
        if value.ndim != 1:
            raise UnsupportedSchemaError(
                f"{dataset.name}: attribute {key} is not a one-dimensional frequency mesh"
            )
        if not np.all(np.isfinite(value)):
            raise UnsupportedSchemaError(
                f"{dataset.name}: attribute {key} contains non-finite values"
            )
        return FrequencyMesh(
            source_path=dataset.name,
            values=value,
            attributes={"mesh_attribute": key, **attrs},
        )
    return None


def _find_meshes(handle: h5py.File) -> tuple[FrequencyMesh, ...]:
    meshes: list[FrequencyMesh] = []

    def visit(name: str, object_: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(object_, h5py.Dataset) or not _path_is_mesh(f"/{name}"):
            return
        attrs = _attributes(object_)
        dtype = object_.dtype
        if dtype.names or dtype.kind not in "iufc":
            raise UnsupportedSchemaError(
                f"/{name}: frequency mesh must be a numeric one-dimensional array"
            )
        values = np.asarray(object_[()])
        if values.ndim != 1:
            raise UnsupportedSchemaError(
                f"/{name}: frequency mesh must be one-dimensional, got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise UnsupportedSchemaError(
                f"/{name}: frequency mesh contains non-finite values"
            )
        meshes.append(FrequencyMesh(f"/{name}", values, attrs))

    handle.visititems(visit)
    return tuple(meshes)


def _explicit_frequency_axis(attrs: dict[str, Any], ndim: int) -> int | None:
    for key in ("frequency_axis", "freq_axis", "mesh_axis"):
        if key not in attrs:
            continue
        value = attrs[key]
        if isinstance(value, str):
            normalized = value.lower()
            if normalized in {"last", "-1"}:
                axis = ndim - 1
            elif normalized in {"first", "0"}:
                axis = 0
            else:
                raise UnsupportedSchemaError(
                    f"unsupported frequency axis label {value!r}"
                )
        else:
            axis = int(value)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise UnsupportedSchemaError(
                f"frequency axis {axis} is outside array dimensions {ndim}"
            )
        return axis
    return None


def _numeric_mesh(value: Any) -> np.ndarray:
    """Convert an attribute mesh to a real or complex numeric NumPy array."""
    raw = np.asarray(value)
    if raw.dtype.kind in "iufc":
        return raw.astype(np.complex128 if raw.dtype.kind == "c" else np.float64)
    if raw.ndim != 1:
        raise UnsupportedSchemaError(
            "attribute-embedded frequency mesh must be one-dimensional"
        )
    try:
        parsed = np.asarray([complex(str(item)) for item in raw], dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise UnsupportedSchemaError(
            "attribute-embedded frequency mesh is not numeric"
        ) from error
    if np.all(parsed.imag == 0):
        return parsed.real
    return parsed


def _associate_mesh(
    source_path: str,
    values: np.ndarray,
    attrs: dict[str, Any],
    meshes: tuple[FrequencyMesh, ...],
) -> tuple[FrequencyMesh | None, int | None]:
    inline_mesh = None
    # Attribute-embedded meshes preserve the candidate dataset as the source path.
    for key in ("matsubara_mesh", "matsubara_frequencies", "frequencies", "omega", "mesh"):
        if key in attrs:
            raw_mesh = _numeric_mesh(attrs[key])
            if raw_mesh.ndim != 1:
                raise UnsupportedSchemaError(
                    f"{source_path}: attribute {key} is not a one-dimensional frequency mesh"
                )
            inline_mesh = FrequencyMesh(
                source_path=source_path,
                values=raw_mesh,
                attributes={"mesh_attribute": key, **attrs},
            )
            break

    mesh = inline_mesh
    if mesh is None:
        if not meshes:
            raise UnsupportedSchemaError(
                f"{source_path}: no verified Matsubara frequency mesh was found"
            )
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            # Multiple mesh datasets are ambiguous unless a source attribute names
            # the exact path.
            mesh_path = attrs.get("frequency_mesh_path") or attrs.get("mesh_path")
            if mesh_path is None:
                raise UnsupportedSchemaError(
                    f"{source_path}: multiple frequency meshes require frequency_mesh_path"
                )
            matches = [item for item in meshes if item.source_path == str(mesh_path)]
            if len(matches) != 1:
                raise UnsupportedSchemaError(
                    f"{source_path}: frequency_mesh_path {mesh_path!r} is not a unique mesh"
                )
            mesh = matches[0]

    if not np.isfinite(mesh.values).all():
        raise UnsupportedSchemaError(f"{source_path}: frequency mesh is not finite")

    axis = _explicit_frequency_axis(attrs, values.ndim)
    matching_axes = [
        index for index, size in enumerate(values.shape) if size == mesh.values.size
    ]
    if axis is not None:
        if values.shape[axis] != mesh.values.size:
            raise UnsupportedSchemaError(
                f"{source_path}: shape {values.shape} does not match mesh length "
                f"{mesh.values.size} on frequency axis {axis}"
            )
        return mesh, axis
    if len(matching_axes) == 1:
        return mesh, matching_axes[0]
    if values.ndim == 0:
        raise UnsupportedSchemaError(
            f"{source_path}: scalar arrays cannot be associated with a frequency mesh"
        )
    if mesh.values.size == values.shape[-1]:
        return mesh, values.ndim - 1
    raise UnsupportedSchemaError(
        f"{source_path}: shape {values.shape} has no unique Matsubara mesh axis "
        f"of length {mesh.values.size}"
    )


def read_hdf5_file(
    file_path: str | Path,
    source_root: str | Path | None = None,
) -> HDF5Extraction:
    """Read and validate candidate G0, G, and Sigma datasets in one HDF5 file."""
    path = Path(file_path)
    root = Path(source_root) if source_root is not None else path.parent
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        handle = h5py.File(path, "r")
    except Exception as error:
        raise UnsupportedSchemaError(f"{path}: cannot open HDF5 file: {error}") from error

    arrays: list[ExtractedArray] = []
    try:
        file_attrs = _attributes(handle)
        meshes = _find_meshes(handle)

        def visit(name: str, object_: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(object_, h5py.Dataset):
                return
            source_path = f"/{name}"
            attrs = _attributes(object_)
            if not _is_candidate(source_path):
                return
            lower_path = source_path.lower()
            imaginary_suffixes = (
                "_imag",
                "_imaginary",
                "/imag",
                "/imaginary",
                ".imag",
                "_i",
                "/i",
            )
            if lower_path.endswith(imaginary_suffixes):
                return
            values, stored_dtype = _reconstruct_complex(handle, object_)
            if values.ndim == 0:
                raise UnsupportedSchemaError(
                    f"{source_path}: candidate array must have at least one dimension"
                )
            mesh, axis = _associate_mesh(source_path, values, attrs, meshes)
            arrays.append(
                ExtractedArray(
                    role=_classify_role(source_path, attrs),
                    source_path=source_path,
                    attributes=attrs,
                    values=values,
                    stored_dtype=stored_dtype,
                    frequency_mesh=mesh,
                    frequency_axis=axis,
                    iteration=_iteration_id(source_path, attrs),
                    spin_labels=_labels(attrs, "spin"),
                    orbital_labels=_labels(attrs, "orbital"),
                )
            )

        handle.visititems(visit)
    finally:
        handle.close()

    return HDF5Extraction(
        source_path=str(path),
        relative_source_path=relative_path,
        sha256=sha256,
        file_attributes=file_attrs,
        arrays=tuple(arrays),
        meshes=meshes,
    )


# Concise public aliases for callers that prefer either extraction or read naming.
extract_hdf5_file = read_hdf5_file
read_hdf5 = read_hdf5_file

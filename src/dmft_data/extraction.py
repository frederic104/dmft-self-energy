"""Representative TRIQS/DFTTools HDF5 extraction.

This module extracts only representations that can be verified from the HDF5
schema.  It deliberately does not infer convergence, chemical parameters,
solver metadata, or physical channel meanings beyond explicit path or attribute
evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .metadata import normalize_run_and_state
from .provenance import sha256_file, software_versions


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


def is_complex_marker(value: Any) -> bool:
    """Interpret the scalar complex marker used by supported HDF5 schemas."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value is True or value == 1


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
    if re.search(r"/g0(?:$|[_/\s-])", text):
        return "dft_derived_g0"
    if "/gimp" in text or "/g_imp" in text or "impurity_green" in text:
        return "impurity_green"
    if "/glat" in text or "/g_lat" in text or "lattice_green" in text:
        return "lattice_green"
    if re.search(r"/g(?:$|[_/-])", text):
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
    if "/g_tau" in name or "/g-tau" in name or re.search(r"/g[^/]*_tau(?:$|/)", name):
        return False
    if _path_is_mesh(name):
        return False
    return (
        re.search(r"/g(?:$|[_./-])", name) is not None
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

    if is_complex_marker(dataset.attrs.get("__complex__", 0)):
        raw = np.asarray(dataset[()])
        if raw.ndim < 1 or raw.shape[-1] != 2:
            raise ComplexRepresentationError(
                f"{dataset.name}: __complex__=1 requires a final Re/Im dimension"
            )
        real = raw[..., 0]
        imag = raw[..., 1]
        if real.dtype.kind not in "iuf" or imag.dtype.kind not in "iuf":
            raise ComplexRepresentationError(
                f"{dataset.name}: BlockGf real and imaginary components "
                "must be numeric"
            )
        values = real + 1j * imag
        if not np.all(np.isfinite(values.real)) or not np.all(
            np.isfinite(values.imag)
        ):
            raise ComplexRepresentationError(
                f"{dataset.name}: reconstructed complex values contain "
                "non-finite numbers"
            )
        return values, "BlockGf real/imaginary last axis"

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
        path = f"/{name}"
        if isinstance(object_, h5py.Dataset):
            if not _path_is_mesh(path):
                return
            if object_.ndim != 1:
                return
            attrs = _attributes(object_)
            dtype = object_.dtype
            if dtype.names or dtype.kind not in "iufc":
                raise UnsupportedSchemaError(
                    f"{path}: frequency mesh must be a numeric one-dimensional array"
                )
            values = np.asarray(object_[()])
            if not np.all(np.isfinite(values)):
                raise UnsupportedSchemaError(
                    f"{path}: frequency mesh contains non-finite values"
                )
            candidate = FrequencyMesh(path, values, attrs)
            if not any(
                np.array_equal(existing.values, candidate.values)
                for existing in meshes
            ):
                meshes.append(candidate)
            return

        if not isinstance(object_, h5py.Group) or not _path_is_mesh(path):
            return
        if "domain" in object_:
            domain = object_["domain"]
        else:
            domain = object_
        if not isinstance(domain, h5py.Group):
            return

        beta_object = domain.get("beta")
        statistic_object = domain.get("statistic")
        size_object = object_.get("size")
        positive_object = object_.get("positive_freq_only")
        if not all(isinstance(item, h5py.Dataset) for item in (
            beta_object,
            statistic_object,
            size_object,
            positive_object,
        )):
            return
        if any(item.shape != () for item in (
            beta_object,
            statistic_object,
            size_object,
            positive_object,
        )):
            return

        beta = float(np.asarray(beta_object[()]).item())
        statistic = np.asarray(statistic_object[()])
        if isinstance(statistic, np.ndarray):
            statistic = statistic.item()
        if isinstance(statistic, bytes):
            statistic = statistic.decode("utf-8", errors="replace")
        size = int(np.asarray(size_object[()]).item())
        positive_only = int(np.asarray(positive_object[()]).item()) == 1

        if beta <= 0 or size < 1:
            raise UnsupportedSchemaError(
                f"{path}: invalid TRIQS MeshImFreq metadata"
            )

        statistic_text = str(statistic).upper()
        fermion = statistic_text.startswith(("F", "FERMI"))
        if positive_only:
            n_iw = size
            indices = np.arange(n_iw, dtype=np.int64)
            total_size = n_iw
        elif fermion:
            n_iw = (size + 1) // 2
            indices = np.arange(-n_iw, n_iw, dtype=np.int64)
            total_size = size
        else:
            n_iw = (size + 1) // 2
            indices = np.arange(-n_iw + 1, n_iw, dtype=np.int64)
            total_size = size

        if indices.size != total_size:
            raise UnsupportedSchemaError(
                f"{path}: mesh size {size} is inconsistent with "
                f"{statistic_text} TRIQS metadata"
            )
        if fermion:
            values = 1j * (2 * indices + 1) * np.pi / beta
        else:
            values = 1j * 2 * indices * np.pi / beta
        if not np.all(np.isfinite(values)):
            raise UnsupportedSchemaError(
                f"{path}: reconstructed frequency mesh contains non-finite values"
            )

        attrs = _attributes(object_)
        attrs.update({
            "mesh_format": "TRIQS MeshImFreq",
            "reconstructed": True,
            "beta": beta,
            "statistic": statistic_text,
            "size": size,
            "positive_freq_only": positive_only,
        })
        candidate = FrequencyMesh(path, values, attrs)
        if not any(
            np.array_equal(existing.values, candidate.values)
            for existing in meshes
        ):
            meshes.append(candidate)

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
            return None, None
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
    sha256 = sha256_file(path)

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
            if object_.dtype.names is None and object_.dtype.kind not in "iufc":
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def write_normalized_hdf5(
    extraction: HDF5Extraction,
    destination: str | Path,
    extraction_config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Write one normalized HDF5 file for a calculation run."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id, physical_state_id = normalize_run_and_state(
        extraction.file_attributes,
        extraction.relative_source_path,
    )
    normalized_keys = {
        key.lower() for key in extraction.file_attributes
    }
    run_id_explicit = bool(normalized_keys & {
        "run_id", "calculation_run_id", "calculation_id", "run",
    }) or bool(extraction.relative_source_path)

    derived_state = None
    beta_values = {
        float(mesh.attributes["beta"])
        for mesh in extraction.meshes
        if "beta" in mesh.attributes
    }
    material = extraction.relative_source_path.split("/", 1)[0]
    if len(beta_values) == 1:
        beta = next(iter(beta_values))
        beta_text = str(int(beta)) if beta.is_integer() else str(beta)
        if material:
            derived_state = f"material={material},beta={beta_text}"
    state_id_explicit = bool(normalized_keys & {
        "physical_state_id", "state_id", "calculation_state_id",
    })
    if not state_id_explicit and derived_state is not None:
        # A material/beta label is useful for review, but does not establish
        # equal U, J, structure, filling, or an independent physical state.
        physical_state_id = "unknown"
    config = extraction_config or {}
    manifest = {
        "source_relative_path": extraction.relative_source_path,
        "source_sha256": extraction.sha256,
        "run_id": run_id,
        "physical_state_id": physical_state_id,
        "extraction_config_sha256": hashlib.sha256(
            _canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "extraction_config": config,
        "software_versions": software_versions(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")

    if output.exists():
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing == manifest and not overwrite:
                return output, {**manifest, "status": "unchanged", "rewritten": False}
            if (
                existing.get("source_sha256") == manifest["source_sha256"]
                and existing.get("source_relative_path")
                == manifest["source_relative_path"]
                and existing.get("extraction_config_sha256")
                == manifest["extraction_config_sha256"]
            ):
                overwrite = True
        if not overwrite:
            raise FileExistsError(
                f"Refusing silent overwrite of existing extraction output: {output}"
            )

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["run_id"] = run_id
            handle.attrs["physical_state_id"] = physical_state_id
            handle.attrs["source_relative_path"] = extraction.relative_source_path
            handle.attrs["source_path"] = extraction.source_path
            handle.attrs["source_sha256"] = extraction.sha256
            handle.attrs["run_id_explicit"] = run_id_explicit
            handle.attrs["physical_state_id_explicit"] = state_id_explicit
            if derived_state is not None:
                handle.attrs["physical_state_candidate"] = derived_state
            if material:
                handle.attrs["material"] = material
            if len(beta_values) == 1:
                beta = next(iter(beta_values))
                handle.attrs["beta"] = beta
                beta_text = str(int(beta)) if beta.is_integer() else str(beta)
                handle.attrs["parameter_block"] = f"beta={beta_text}"
            handle.attrs["extraction_config"] = _canonical_json(config)
            handle.attrs["h5py_version"] = software_versions()["h5py"]
            handle.attrs["numpy_version"] = software_versions()["numpy"]

            run_group = handle.create_group("run")
            run_group.attrs["run_id"] = run_id
            run_group.attrs["physical_state_id"] = physical_state_id
            run_group.attrs["source_relative_path"] = extraction.relative_source_path

            provenance = run_group.create_group("provenance")
            provenance.attrs["source_path"] = extraction.source_path
            provenance.attrs["relative_source_path"] = extraction.relative_source_path
            provenance.attrs["sha256"] = extraction.sha256

            arrays_group = run_group.create_group("arrays")
            for index, array in enumerate(extraction.arrays):
                dataset = arrays_group.create_dataset(
                    f"array_{index:04d}",
                    data=np.asarray(array.values),
                )
                dataset.attrs["source_path"] = array.source_path
                dataset.attrs["role"] = array.role
                dataset.attrs["stored_dtype"] = array.stored_dtype
                dataset.attrs["frequency_axis"] = (
                    -1 if array.frequency_axis is None else array.frequency_axis
                )
                dataset.attrs["iteration"] = array.iteration or "unknown"
                for key, value in array.attributes.items():
                    try:
                        dataset.attrs[f"source_attribute_{key}"] = _json_attribute(value)
                    except (TypeError, ValueError):
                        dataset.attrs[f"source_attribute_{key}"] = str(value)
                if array.spin_labels is not None:
                    dataset.attrs["spin_labels"] = array.spin_labels
                if array.orbital_labels is not None:
                    dataset.attrs["orbital_labels"] = array.orbital_labels
                if array.frequency_mesh is not None:
                    mesh = array.frequency_mesh
                    mesh_dataset = arrays_group.create_dataset(
                        f"mesh_{index:04d}",
                        data=np.asarray(mesh.values),
                    )
                    mesh_dataset.attrs["source_path"] = mesh.source_path
                    for key, value in mesh.attributes.items():
                        try:
                            mesh_dataset.attrs[f"source_attribute_{key}"] = _json_attribute(value)
                        except (TypeError, ValueError):
                            mesh_dataset.attrs[f"source_attribute_{key}"] = str(value)

            meshes_group = run_group.create_group("meshes")
            for index, mesh in enumerate(extraction.meshes):
                mesh_dataset = meshes_group.create_dataset(
                    f"mesh_{index:04d}",
                    data=np.asarray(mesh.values),
                )
                mesh_dataset.attrs["source_path"] = mesh.source_path
                for key, value in mesh.attributes.items():
                    try:
                        mesh_dataset.attrs[f"source_attribute_{key}"] = _json_attribute(value)
                    except (TypeError, ValueError):
                        mesh_dataset.attrs[f"source_attribute_{key}"] = str(value)

        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest_path.write_text(
        _canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return output, {**manifest, "status": "written", "rewritten": bool(overwrite)}


def _json_attribute(value: Any) -> Any:
    return _json_value(value)


def run_extraction(
    config_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Extract configured HDF5 sources into normalized per-run HDF5 files."""
    config_path = Path(config_path)
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.resolve().parent.parent
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")

    paths = config["paths"]
    raw_dir = root / paths["raw_dir"]
    processed_dir = root / paths["processed_dir"]
    report_dir = root / paths["report_dir"]
    extensions = config["dataset"].get("file_extensions", [".h5", ".hdf5"])
    recursive = config["dataset"].get("recursive", True)
    output_root = processed_dir / "arrays"

    normalized_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    if raw_dir.is_dir():
        candidates = raw_dir.rglob("*") if recursive else raw_dir.iterdir()
        sources = sorted(
            (path for path in candidates if path.is_file()
             and path.suffix.lower() in normalized_extensions),
            key=lambda path: path.as_posix(),
        )
    else:
        sources = []

    report: dict[str, Any] = {
        "status": "passed" if raw_dir.is_dir() else "failed",
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_root.resolve()),
        "extraction_config": config.get("extraction", {}),
        "software_versions": software_versions(),
        "files": [],
        "failures": [],
    }
    if not raw_dir.is_dir():
        report["failures"].append({
            "relative_path": str(Path(paths["raw_dir"]).as_posix()),
            "source_path": str(raw_dir.resolve()),
            "error_type": "FileNotFoundError",
            "error": f"Configured raw data directory does not exist: {raw_dir}",
        })

    extraction_config = config.get("extraction", {})
    for source in sources:
        relative = source.relative_to(raw_dir).as_posix()
        try:
            extraction = read_hdf5_file(source, raw_dir)
            relative_output = Path(relative)
            output = output_root / relative_output.with_suffix(".h5")
            _, result = write_normalized_hdf5(
                extraction,
                output,
                extraction_config=extraction_config,
            )
            report["files"].append({
                "relative_source_path": relative,
                "source_sha256": extraction.sha256,
                "output_path": str(output.resolve()),
                "status": result["status"],
                "array_count": len(extraction.arrays),
            })
        except Exception as error:  # noqa: BLE001
            report["status"] = "failed"
            report["failures"].append({
                "relative_path": relative,
                "source_path": str(source.resolve()),
                "source_sha256": sha256_file(source),
                "error_type": type(error).__name__,
                "error": str(error),
            })

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "hdf5_extraction.json").write_text(
        _canonical_json(report) + "\n",
        encoding="utf-8",
    )
    return report

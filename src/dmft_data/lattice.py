"""Reference lattice Dyson calculation in one explicit orthonormal spin block.

Inputs must already share basis, units, energy zero and correlated subspace.
This module does not convert HDF5, perform continuation or certify that contract.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from .provenance import sha256_file


def _dagger(a: np.ndarray) -> np.ndarray:
    return a.swapaxes(-1, -2).conj()


def lattice_green(hopping: np.ndarray, projectors: np.ndarray, sigma: np.ndarray,
                  z: np.ndarray, mu: float, dc: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate G_k=[(z+mu)I-H_k-P_k^dag(Sigma-dc)P_k]^-1.

    H: (k, band, band); P: (k, correlated, band);
    Sigma: (frequency, correlated, correlated); dc: (correlated, correlated).
    Sigma is the impurity self-energy BEFORE subtracting dc. z is relative to mu.
    One block is processed without implicit spin multiplicity or orbital rotation.
    """
    h, p, s, z, dc = [np.asarray(a, dtype=complex) for a in
                      (hopping, projectors, sigma, z, dc)]
    if (h.ndim != 3 or p.ndim != 3 or s.ndim != 3 or z.ndim != 1
            or min(h.shape) == 0 or min(p.shape) == 0 or len(z) == 0
            or h.shape[1] != h.shape[2] or p.shape[0] != h.shape[0]
            or p.shape[2] != h.shape[1] or s.shape != (len(z), p.shape[1], p.shape[1])
            or dc.shape != s.shape[1:]):
        raise ValueError("Invalid k/band/correlated/frequency dimensions")
    if (not all(np.isfinite(a).all() for a in (h, p, s, z, dc))
            or np.ndim(mu) != 0 or np.iscomplexobj(mu) or not np.isfinite(mu)
            or np.any(z.imag <= 0)):
        raise ValueError("Finite inputs, real scalar mu and upper-half-plane z required")
    if not np.allclose(h, _dagger(h), rtol=1e-10, atol=1e-12):
        raise ValueError("Hopping must be Hermitian in an orthonormal basis")
    if not np.allclose(dc, _dagger(dc), rtol=1e-10, atol=1e-12):
        raise ValueError("DC potential must be a Hermitian matrix, not dc_energ")
    gk = np.empty((len(h), len(z), h.shape[1], h.shape[2]), dtype=complex)
    projected = np.empty((len(h), *s.shape), dtype=complex)
    identity = np.eye(h.shape[1])
    for k in range(len(h)):
        inverse = ((z + mu)[:, None, None] * identity - h[k]
                   - _dagger(p[k]) @ (s - dc) @ p[k])
        gk[k] = np.linalg.inv(inverse)
        projected[k] = p[k] @ gk[k] @ _dagger(p[k])
    if not np.isfinite(gk).all() or not np.isfinite(projected).all():
        raise ValueError("Non-finite lattice Green function")
    return {"green_k": gk, "projected_green_k": projected}


def bz_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average an actual BZ quadrature; never renormalize weights silently."""
    w = np.asarray(weights)
    if (w.shape != (len(values),) or np.iscomplexobj(w)
            or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1., rtol=1e-10, atol=1e-12)):
        raise ValueError("BZ weights must be finite, nonnegative and sum to one")
    return np.tensordot(w, values, axes=(0, 0))


def process(arrays: dict, conventions: dict) -> dict[str, np.ndarray]:
    """Apply declared conventions; declarations do not verify a material archive."""
    required = {"basis": "orthonormal", "sigma_convention": "impurity_before_dc",
                "energy_axis": "relative_to_mu", "spin": "one_explicit_block"}
    if any(conventions.get(k) != v for k, v in required.items()):
        raise ValueError("Explicit orthonormal/before-DC/relative-mu/spin contract required")
    if conventions.get("energy_unit") not in {"eV", "model_units"}:
        raise ValueError("Declare a common energy_unit: eV or model_units")
    if conventions.get("interpretation") not in {"synthetic", "exploratory_material"}:
        raise ValueError("Interpretation must be synthetic or exploratory_material")
    sampling, domain = conventions.get("sampling"), conventions.get("domain")
    if sampling not in {"bz", "path"} or domain not in {"matsubara", "real_frequency"}:
        raise ValueError("Declare sampling (bz/path) and domain (matsubara/real_frequency)")
    z = np.asarray(arrays["z"])
    result = lattice_green(*(arrays[k] for k in ("hopping", "projectors", "sigma", "z", "mu", "dc")))
    if domain == "matsubara":
        if not np.allclose(z.real, 0, atol=1e-14) or np.any(np.diff(z.imag) <= 0):
            raise ValueError("Matsubara mesh must be imaginary and ordered")
    elif np.any(np.diff(z.real) <= 0) or not np.allclose(z.imag, z.imag[0]):
        raise ValueError("Real-frequency grid must be ordered with constant positive eta")
    if sampling == "path":
        if "weights" in arrays:
            raise ValueError("A k path is not a BZ quadrature; omit weights")
        distance = np.asarray(arrays["k_distance"])
        if (distance.shape != (len(result["green_k"]),) or np.iscomplexobj(distance)
                or not np.isfinite(distance).all() or np.any(np.diff(distance) <= 0)):
            raise ValueError("Provide strictly increasing path distance for every k")
        result["k_distance"] = distance
    else:
        result["green_local"] = bz_average(result["projected_green_k"], arrays["weights"])
    if domain == "real_frequency":
        result["spectral_k"] = -np.trace(result["green_k"], axis1=-2, axis2=-1).imag / np.pi
        result["projected_spectral_k"] = -np.trace(result["projected_green_k"], axis1=-2, axis2=-1).imag / np.pi
        if sampling == "bz":
            result["dos"] = bz_average(result["spectral_k"], arrays["weights"])
            result["projected_dos"] = bz_average(result["projected_spectral_k"], arrays["weights"])
    return {"z": z, **result}


def run(input_path: Path, config: Path, output: Path) -> dict:
    """Process normalized NPZ data; preserve source hashes without local roots."""
    if output.exists():
        raise FileExistsError(output)
    conventions = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(conventions, dict):
        raise TypeError("Expected convention mapping")
    with np.load(input_path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    result = process(arrays, conventions)
    output.mkdir(parents=True)
    np.savez_compressed(output / "lattice.npz", **result)
    # Only the declared vocabulary is shared, not arbitrary local config metadata.
    keys = ("basis", "sigma_convention", "energy_axis", "spin", "energy_unit",
            "interpretation", "sampling", "domain")
    summary = {"conventions": {k: conventions[k] for k in keys},
               "input_sha256": sha256_file(input_path), "config_sha256": sha256_file(config),
               "output_sha256": sha256_file(output / "lattice.npz"),
               "source_sha256": sha256_file(Path(__file__)),
               "versions": {"python": platform.python_version(), "numpy": np.__version__},
               "shape_green_k": list(result["green_k"].shape),
               "material_validation": "not_assessed",
               "note": "Fixed-mu reference postprocessing; no continuation, DFT, DMFT loop, filling adjustment, basis matching or causality certification. No spin factor, clipping or weight renormalization."}
    if "spectral_k" in result:
        summary["negative_spectral_trace_count"] = int(np.sum(result["spectral_k"] < -1e-10))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run a normalized reference calculation without guessing HDF5 conventions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.config, args.output)


if __name__ == "__main__":
    main()

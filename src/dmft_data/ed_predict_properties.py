"""Reconstruct finite-bath scalar AIM properties from supplied Sigma predictions."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .ed_properties import dyson_green, occupancy_estimates
from .provenance import sha256_file


def reconstruct(g0: np.ndarray, sigma: np.ndarray, omega: np.ndarray,
                beta: float) -> dict[str, np.ndarray]:
    """Return per-spin impurity G and occupation estimates without reference data.

    The caller must supply a scalar AIM noninteracting impurity propagator and
    matching Sigma for each physical state on the same positive-frequency mesh.
    This function cannot verify those physical semantics or model accuracy.
    """
    if (g0.ndim != 2 or g0.shape[0] == 0 or sigma.shape != g0.shape
            or omega.ndim != 1 or np.iscomplexobj(omega) or g0.shape[1] != len(omega)
            or not np.iscomplexobj(g0) or not np.iscomplexobj(sigma)):
        raise ValueError("Expected nonempty complex state-by-frequency G0 and Sigma")
    green = dyson_green(g0, sigma)
    raw, tail = occupancy_estimates(green, omega, beta)
    return {"green": green, "occupancy_raw": raw, "occupancy_tail": tail}


def run(input_path: Path, output: Path) -> dict:
    """Read a declared scalar AIM NPZ and write predictions with provenance."""
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    digest = sha256_file(input_path)
    with np.load(input_path, allow_pickle=False) as data:
        required = {"g0", "sigma", "omega", "beta"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing required arrays: {', '.join(sorted(missing))}")
        g0, sigma, omega, beta = (data[k].copy() for k in ("g0", "sigma", "omega", "beta"))
    if beta.ndim != 0 or np.iscomplexobj(beta):
        raise ValueError("beta must be a real scalar shared by all states")
    beta = float(beta)
    values = reconstruct(g0, sigma, omega, beta)
    output.mkdir(parents=True)
    np.savez_compressed(output / "properties.npz", omega=omega, beta=beta, **values)
    summary = {
        "scope": "scalar_finite_bath_AIM_impurity", "input_sha256": digest,
        "input_label": input_path.name, "state_count": len(g0),
        "frequency_count": len(omega), "beta": beta,
        "outputs_sha256": sha256_file(output / "properties.npz"),
        "source_sha256": {p.name: sha256_file(p) for p in (
            Path(__file__), Path(__file__).with_name("ed_properties.py"))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "occupancy_tail_out_of_bounds_count": int(np.sum(
            (values["occupancy_tail"] < 0) | (values["occupancy_tail"] > 1))),
        "positive_imaginary_green_count": int(np.sum(values["green"].imag > 1e-10)),
        "limitations": [
            "No model inference, reference comparison or accuracy estimate was performed.",
            "G0 must be the matching scalar AIM impurity propagator; meanings and state order are supplied by the caller.",
            "Tail-corrected per-spin occupation is an approximation and is not clipped.",
            "Matsubara impurity output is not a real-frequency spectrum or lattice DOS.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run the reference-free AIM property reconstruction command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(f"Reconstructed {result['state_count']} AIM states in {args.output}")


if __name__ == "__main__":
    main()

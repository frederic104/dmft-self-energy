"""Compare saved finite-bath impurity spectra for one physical state."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .provenance import sha256_file


def visualize(source: Path, output: Path, state: int) -> dict:
    """Plot available saved predictions against the matched broadened ED spectrum."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as data:
        energy = data["energy"].copy()
        reference = data["reference_spectrum"].copy()
        parameters = data["parameters"].copy()
        eta = float(data["eta"])
        keys = sorted(k for k in data.files if k.startswith("seed_") and k.endswith("_spectrum"))
        predictions = {k: data[k].copy() for k in keys}
    if (energy.ndim != 1 or len(energy) < 2 or not np.isfinite(energy).all()
            or np.any(np.diff(energy) <= 0) or not np.isfinite(eta) or eta <= 0
            or reference.ndim != 2 or reference.shape[1] != len(energy)
            or len(parameters) != len(reference) or not 0 <= state < len(reference)
            or not np.isfinite(reference).all() or not keys):
        raise ValueError("Invalid spectral data, state, grid or missing predictions")
    if any(a.shape != reference.shape or not np.isfinite(a).all() for a in predictions.values()):
        raise ValueError("Invalid spectrum prediction")
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(energy, reference[state], color="black", label="ED reference")
    for key, values in predictions.items():
        ax.plot(energy, values[state], label=key.removesuffix("_spectrum"), alpha=.75)
    ax.set(xlabel="Energy relative to chemical potential (native units)",
           ylabel="Impurity spectrum per spin", title=f"State {state}; broadening eta={eta:g}")
    ax.legend(fontsize="small")
    output.mkdir(parents=True)
    fig.savefig(output / "impurity_spectrum.png", dpi=160)
    summary = {"input_label": source.name, "input_sha256": sha256_file(source),
               "state_index": state, "parameters": parameters[state].tolist(),
               "eta": eta, "predictions": keys,
               "scope": "Broadened finite-bath impurity spectrum; not lattice DOS. Seeds share the same state. No new inference or model selection."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Render one saved spectrum file without loading models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = visualize(args.input, args.output, args.state)
    print(f"Saved state {summary['state_index']} with {len(summary['predictions'])} prediction curves")


if __name__ == "__main__":
    main()

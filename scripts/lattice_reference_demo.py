"""Synthetic lattice postprocessing example; no DFT, DMFT solve or ML prediction."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dmft_data.lattice import run
from dmft_data.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    z = np.linspace(-4, 4, 801) + .08j
    # Chosen analytic causal self-energy, not inferred from Matsubara data.
    sigma = (.4 + .3 / (z - 1.5))[:, None, None]
    convention = {"basis": "orthonormal", "sigma_convention": "impurity_before_dc",
                  "energy_axis": "relative_to_mu", "spin": "one_explicit_block",
                  "energy_unit": "model_units", "interpretation": "synthetic",
                  "domain": "real_frequency"}
    for name in ("bz", "path", "noninteracting"):
        sampling = "path" if name == "path" else "bz"
        k = (np.linspace(0, np.pi, 121) if sampling == "path"
             else np.linspace(-np.pi, np.pi, 256, endpoint=False))
        arrays = {"hopping": (-2*np.cos(k))[:, None, None],
                  "projectors": np.ones((len(k), 1, 1)), "z": z, "mu": np.array(0.),
                  "sigma": sigma, "dc": np.array([[.4]])}
        if sampling == "path":
            arrays["k_distance"] = k
        else:
            arrays["weights"] = np.full(len(k), 1/len(k))
        if name == "noninteracting":
            arrays.update(sigma=np.zeros_like(sigma), dc=np.zeros((1, 1)))
        source, cfg = out/f"{name}_input.npz", out/f"{name}.yaml"
        np.savez_compressed(source, **arrays)
        cfg.write_text(yaml.safe_dump(dict(convention, sampling=sampling)), encoding="utf-8")
        run(source, cfg, out/name)

    with np.load(out/"bz/lattice.npz", allow_pickle=False) as saved:
        dos = saved["dos"]
    with np.load(out/"noninteracting/lattice.npz", allow_pickle=False) as saved:
        dos0 = saved["dos"]
    with np.load(out/"path/lattice.npz", allow_pickle=False) as saved:
        distance, ak = saved["k_distance"], saved["spectral_k"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(z.real, dos0, label="Noninteracting chain")
    axes[0].plot(z.real, dos, label="Chosen causal self-energy")
    axes[0].set(xlabel="Energy relative to mu (model units)",
                ylabel="DOS per spin block (1 / model unit)")
    axes[0].legend()
    mesh = axes[1].pcolormesh(distance, z.real, ak.T, shading="auto", cmap="magma")
    axes[1].set(xlabel="k (lattice constant = 1)",
                ylabel="Energy relative to mu (model units)",
                xticks=[0, np.pi], xticklabels=["0", "pi"])
    fig.colorbar(mesh, ax=axes[1], label="A(k, energy) per spin block")
    fig.suptitle("Synthetic chain; fixed mu = 0; eta = 0.08; no material inference")
    figure = out/"lattice_demo.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {"scope": "Synthetic one-band chain with chosen rational self-energy; no learned or material result.",
               "parameters": {"hopping_t": 1., "sigma_static": .4, "pole": 1.5,
                              "residue": .3, "dc": .4, "mu": 0., "eta": .08},
               "generator_sha256": sha256_file(Path(__file__)),
               "figure_sha256": sha256_file(figure),
               "matplotlib_version": plt.matplotlib.__version__,
               "results": {name: sha256_file(out/name/"summary.json")
                           for name in ("bz", "path", "noninteracting")}}
    (out/"demo.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved synthetic DOS and path spectrum; no DFT/DMFT/ML run.")


if __name__ == "__main__":
    main()

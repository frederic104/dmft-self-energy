"""Plot exploratory AIM test errors and the worst test curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(result_dir: Path) -> Path:
    """Write one diagnostic figure without overwriting previous figures."""
    result_dir = Path(result_dir)
    output = result_dir / "test_diagnostics.png"
    if output.exists():
        raise FileExistsError(output)
    with np.load(result_dir / "test_predictions.npz", allow_pickle=False) as data:
        params, omega, ref, pred, indices = (data[key].copy() for key in (
            "parameters", "omega", "reference", "prediction", "indices"
        ))
    errors = np.sqrt(np.mean(np.abs(pred[:, :8] - ref[:, :8]) ** 2, axis=1))
    worst = int(np.argmax(errors))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    scatter = axes[0].scatter(params[:, 0], params[:, 1], c=errors, cmap="viridis")
    fig.colorbar(scatter, ax=axes[0], label="first 8 complex RMSE")
    axes[0].set(xlabel="epsilon_d", ylabel="U", title=f"Test states (n={len(params)})")
    for ax, part, label in zip(axes[1:], ("real", "imag"), ("Re Sigma", "Im Sigma")):
        ax.plot(omega, getattr(ref[worst], part), label="reference")
        ax.plot(omega, getattr(pred[worst], part), "--", label="prediction")
        ax.set(xlabel="Matsubara frequency", ylabel=label,
               title=f"Worst first-8 error: sample {indices[worst]}")
        ax.legend()
    fig.suptitle("Exploratory AIM test diagnostics; scatter shows individual states")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    print(plot(parser.parse_args().result))


if __name__ == "__main__":
    main()

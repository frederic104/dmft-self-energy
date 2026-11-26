"""Exploratory fixed-U AIM baselines with impurity-level blocked holdouts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .learning.models import Baseline, decode, encode
from .learning.splits import blocked_partition
from .provenance import sha256_file


def run(dataset: Path, output: Path, seed: int = 0) -> dict:
    """Fit fixed-budget baselines, select on validation, evaluate test once."""
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    with np.load(dataset, allow_pickle=False) as data:
        columns = data["parameter_columns"].tolist()
        params, g0, sigma, omega = (data[k].copy() for k in (
            "parameters", "g0", "sigma", "omega"
        ))
    expected = ["epsilon_d", "interaction_u", "epsilon_b0", "epsilon_b1", "V_b0",
                "V_b1", "chemical_potential", "beta"]
    if columns != expected or params.ndim != 2 or params.shape[1] != 8 or not len(params):
        raise ValueError("Unexpected parameter schema")
    if (g0.shape != sigma.shape or g0.shape != (len(params), len(omega))
            or not all(np.isfinite(a).all() for a in (params, g0, sigma, omega))
            or np.any(omega <= 0)):
        raise ValueError("Invalid arrays or Matsubara mesh")
    if not all(np.all(params[:, i] == params[0, i]) for i in (1, 6, 7)):
        raise ValueError("U, mu and beta must be fixed")
    expected_omega = (2 * np.arange(len(omega)) + 1) * np.pi / params[0, 7]
    if params[0, 7] <= 0 or not np.allclose(omega, expected_omega):
        raise ValueError("Mesh must match the fixed beta")
    records = []
    for i, row in enumerate(params):
        bath = sorted(((row[2], abs(row[4])), (row[3], abs(row[5]))))
        group = repr((row[0], row[1], bath, row[6], row[7]))
        block = "low" if row[0] < -0.6 else "high" if row[0] >= -0.2 else "middle"
        records.append({"group_id": group, "parameter_block": block,
                        "relative_source_path": str(i)})
    if len({r["group_id"] for r in records}) != len(records):
        raise ValueError("Duplicate physical parameters; deduplicate before fitting")
    splits = blocked_partition(records, {"dimension": "parameter_block",
                                         "validation": ["low"], "test": ["high"]})
    x, y = encode(g0), encode(sigma)
    candidates = [{"name": "mean"}, {"name": "nearest"},
                  {"name": "pca_ridge", "latent_dimension": 4}, {"name": "mlp"}]
    trained, validation = {}, {}
    for spec in candidates:
        name = spec["name"]
        model = Baseline(spec, seed).fit(x[splits["train"]], y[splits["train"]])
        prediction = decode(model.predict(x[splits["validation"]]), sigma.shape[1:])
        validation[name] = float(np.sqrt(np.mean(np.abs(
            prediction - sigma[splits["validation"]]) ** 2)))
        trained[name] = model
    selected = min(validation, key=validation.get)
    prediction = decode(trained[selected].predict(x[splits["test"]]), sigma.shape[1:])
    target = sigma[splits["test"]]
    difference = prediction - target
    metrics = {
        "dataset_sha256": sha256_file(dataset),
        "seed": seed, "python": platform.python_version(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "candidates": candidates,
        "split_rule": "epsilon_d: validation < -0.6; train [-0.6,-0.2); test >= -0.2",
        "splits": splits, "validation_complex_rmse": validation, "selected": selected,
        "test_complex_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "test_first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(difference[:, :8]) ** 2))),
        "note": "Finite-bath AIM only; held-out epsilon_d blocks; not lattice DMFT.",
    }
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump(trained[selected], output / "selected_model.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()

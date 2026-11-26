"""Exploratory fixed-U AIM baselines with impurity-level blocked holdouts."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .learning.models import Baseline, decode, encode
from .learning.splits import blocked_partition
from .provenance import sha256_file


def run(dataset: Path, output: Path, seed: int = 0, split: str = "epsilon",
        include_latent: bool = False) -> dict:
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
    if split not in {"epsilon", "random", "u"}:
        raise ValueError("split must be epsilon, random, or u")
    if not all(np.all(params[:, i] == params[0, i]) for i in (6, 7)):
        raise ValueError("mu and beta must be fixed")
    if split != "u" and not np.all(params[:, 1] == params[0, 1]):
        raise ValueError("U must be fixed for epsilon and random splits")
    expected_omega = (2 * np.arange(len(omega)) + 1) * np.pi / params[0, 7]
    if params[0, 7] <= 0 or not np.allclose(omega, expected_omega):
        raise ValueError("Mesh must match the fixed beta")
    records = []
    for i, row in enumerate(params):
        bath = sorted(((row[2], abs(row[4])), (row[3], abs(row[5]))))
        group = repr((row[0], row[1], bath, row[6], row[7]))
        if split == "u":
            block = "validation" if 1.5 <= row[1] < 2.0 else (
                "test" if 2.0 <= row[1] < 2.5 else "train"
            )
        else:
            block = "low" if row[0] < -0.6 else "high" if row[0] >= -0.2 else "middle"
        records.append({"group_id": group, "parameter_block": block,
                        "relative_source_path": str(i)})
    if len({r["group_id"] for r in records}) != len(records):
        raise ValueError("Duplicate physical parameters; deduplicate before fitting")
    if split == "random":
        if len(params) < 10:
            raise ValueError("Random split requires at least 10 distinct states")
        order = np.random.default_rng(seed).permutation(len(params))
        n_test = max(1, int(len(order) * 0.2))
        n_val = max(1, int(len(order) * 0.2))
        splits = {"test": order[:n_test].tolist(),
                  "validation": order[n_test:n_test + n_val].tolist(),
                  "train": order[n_test + n_val:].tolist()}
        rule = "random physical-state split: 60% train, 20% validation, 20% test"
    else:
        validation_block, test_block = ("validation", "test") if split == "u" else ("low", "high")
        splits = blocked_partition(records, {"dimension": "parameter_block",
                                             "validation": [validation_block],
                                             "test": [test_block]})
        rule = ("U: validation [1.5,2); test [2,2.5); train remainder" if split == "u"
                else "epsilon_d: validation < -0.6; train [-0.6,-0.2); test >= -0.2")
    x, y = encode(g0), encode(sigma)
    if split == "u":
        x = np.column_stack((x, params[:, 1]))
    candidates = [{"name": "mean"}, {"name": "nearest"},
                  {"name": "pca_ridge", "latent_dimension": 4}, {"name": "mlp"}]
    if include_latent:
        candidates = [{"name": "mean"}, {"name": "nearest"}, {"name": "mlp"}]
        candidates += [{"name": name, "latent_dimension": dim}
                       for name in ("pca_ridge", "latent_mlp") for dim in (4, 8, 16)]
    trained, validation, fit_warnings, validation_details = {}, {}, {}, {}
    for spec in candidates:
        name = spec["name"]
        if include_latent and "latent_dimension" in spec:
            name = f"{name}_{spec['latent_dimension']}"
            if spec["name"] == "pca_ridge" and len(splits["train"]) <= spec["latent_dimension"]:
                raise ValueError("Training set must exceed the requested PCA dimension")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Baseline(spec, seed).fit(x[splits["train"]], y[splits["train"]])
        fit_warnings[name] = [str(item.message) for item in caught]
        prediction = decode(model.predict(x[splits["validation"]]), sigma.shape[1:])
        validation[name] = float(np.sqrt(np.mean(np.abs(
            prediction - sigma[splits["validation"]]) ** 2)))
        delta = prediction - sigma[splits["validation"]]
        validation_details[name] = {
            "complex_rmse": validation[name],
            "real_rmse": float(np.sqrt(np.mean(delta.real ** 2))),
            "imaginary_rmse": float(np.sqrt(np.mean(delta.imag ** 2))),
            "first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(delta[:, :8]) ** 2))),
            "last_8_complex_rmse": float(np.sqrt(np.mean(np.abs(delta[:, -8:]) ** 2))),
            "positive_imaginary_fraction": float(np.mean(prediction.imag > 1e-10)),
            "actual_latent_dimension": getattr(model, "actual_dimension", None),
        }
        trained[name] = model
    selected = min(validation, key=validation.get)
    prediction = decode(trained[selected].predict(x[splits["test"]]), sigma.shape[1:])
    target = sigma[splits["test"]]
    difference = prediction - target
    metrics = {
        "dataset_sha256": sha256_file(dataset),
        "seed": seed, "python": platform.python_version(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "candidates": candidates,
        "split_rule": rule, "split_type": split, "input_includes_u": split == "u",
        "feature_order": "flattened G0 real, G0 imaginary" + (", U" if split == "u" else ""),
        "fit_warnings": fit_warnings,
        "validation_diagnostics": validation_details,
        "splits": splits, "validation_complex_rmse": validation, "selected": selected,
        "test_complex_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "test_first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(difference[:, :8]) ** 2))),
        "test_last_8_complex_rmse": float(np.sqrt(np.mean(np.abs(difference[:, -8:]) ** 2))),
        "test_real_rmse": float(np.sqrt(np.mean(difference.real ** 2))),
        "test_imaginary_rmse": float(np.sqrt(np.mean(difference.imag ** 2))),
        "test_positive_imaginary_fraction": float(np.mean(prediction.imag > 1e-10)),
        "reference_positive_imaginary_fraction": float(np.mean(target.imag > 1e-10)),
        "note": "Exploratory finite-bath AIM; not lattice DMFT or a final test.",
    }
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump(trained[selected], output / "selected_model.joblib")
    np.savez_compressed(output / "test_predictions.npz", indices=np.asarray(splits["test"]),
                        parameters=params[splits["test"]], omega=omega,
                        reference=target, prediction=prediction)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("epsilon", "random", "u"), default="epsilon")
    parser.add_argument("--include-latent", action="store_true",
                        help="Compare PCA and unconstrained MLP bottlenecks 4, 8, 16")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.seed, args.split,
                         args.include_latent), indent=2))


if __name__ == "__main__":
    main()

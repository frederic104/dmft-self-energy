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

from .learning.models import Baseline, EpsilonLinear, decode, encode
from .learning.splits import blocked_partition
from .provenance import sha256_file


def _network_diagnostics(model: Baseline, train_x: np.ndarray, val_x: np.ndarray) -> dict:
    """Summarize standardized-target loss and observed bottleneck activations."""
    network = model.model
    result = {
        "loss_curve": [float(value) for value in network.loss_curve_],
        "loss_definition": "sklearn MLP training objective on standardized targets, including regularization",
        "max_iter": network.max_iter,
        "solver": network.solver,
        "learning_rate_init": network.learning_rate_init,
        "batch_size": network.batch_size,
        "alpha": network.alpha,
        "activation": network.activation,
        "tolerance": network.tol,
        "n_iter_no_change": network.n_iter_no_change,
        "early_stopping": network.early_stopping,
    }
    if model.specification["name"] == "latent_mlp":
        result["bottleneck"] = {}
        for role, values in (("train", train_x), ("validation", val_x)):
            hidden = model.x_scaler.transform(values)
            # The second hidden layer is the bottleneck in (64, d, 64).
            for layer in range(2):
                hidden = hidden @ network.coefs_[layer] + network.intercepts_[layer]
                hidden = np.maximum(hidden, 0) if network.activation == "relu" else np.tanh(hidden)
            result["bottleneck"][role] = {
                "units": int(hidden.shape[1]),
                "positive_activation_fraction": float(np.mean(hidden > 0)),
                "mean_per_unit_standard_deviation": float(np.mean(np.std(hidden, axis=0))),
            }
            if network.activation == "relu":
                result["bottleneck"][role].update({
                    "unit_never_positive_count": int(np.sum(np.all(hidden == 0, axis=0))),
                    "all_units_zero_sample_fraction": float(np.mean(np.all(hidden == 0, axis=1))),
                })
            else:
                result["bottleneck"][role]["saturated_fraction_abs_ge_099"] = float(
                    np.mean(np.abs(hidden) >= 0.99)
                )
    return result


def run(dataset: Path, output: Path, seed: int = 0, split: str = "epsilon",
        include_latent: bool = False, model_seed: int | None = None,
        compare_activations: bool = False, validation_only: bool = False,
        include_epsilon_linear: bool = False, save_candidates: bool = False) -> dict:
    """Fit fixed-budget baselines, select on validation, evaluate test once."""
    dataset, output = Path(dataset), Path(output)
    model_seed = seed if model_seed is None else model_seed
    if compare_activations and not include_latent:
        raise ValueError("Activation comparison requires include_latent")
    if save_candidates and not validation_only:
        raise ValueError("Saving candidate bundle requires validation_only")
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
    if include_epsilon_linear and (split == "u" or
            not np.all(params[:, 1:] == params[0, 1:])):
        raise ValueError("epsilon linear baseline requires fixed U, bath, mu and beta")
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
    if include_latent and (len(splits["train"]) <= 16 or y.shape[1] < 16):
        raise ValueError("Training samples and target features must support PCA dimension 16")
    candidates = [{"name": "mean"}, {"name": "nearest"},
                  {"name": "pca_ridge", "latent_dimension": 4}, {"name": "mlp"}]
    if include_latent:
        candidates = [{"name": "mean"}, {"name": "nearest"}, {"name": "mlp"}]
        candidates += [{"name": name, "latent_dimension": dim}
                       for name in ("pca_ridge", "latent_mlp") for dim in (1, 2, 4, 8, 16)]
    if compare_activations:
        candidates += [{"name": "mlp", "activation": "tanh"}]
        candidates += [{"name": "latent_mlp", "activation": "tanh", "latent_dimension": dim}
                       for dim in (1, 2, 4, 8, 16)]
    if include_epsilon_linear:
        candidates.append({"name": "epsilon_linear", "input": "epsilon_d"})
    trained, validation, fit_warnings, validation_details = {}, {}, {}, {}
    network_diagnostics = {}
    for spec in candidates:
        name = spec["name"]
        if spec.get("activation") == "tanh":
            name += "_tanh"
        if include_latent and "latent_dimension" in spec:
            name = f"{name}_{spec['latent_dimension']}"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = (EpsilonLinear().fit(params[splits["train"], 0], y[splits["train"]])
                     if name == "epsilon_linear" else
                     Baseline(spec, model_seed).fit(x[splits["train"]], y[splits["train"]]))
        fit_warnings[name] = [str(item.message) for item in caught]
        if spec["name"] in {"mlp", "latent_mlp"}:
            network_diagnostics[name] = _network_diagnostics(
                model, x[splits["train"]], x[splits["validation"]]
            )
        val_input = params[splits["validation"], 0] if name == "epsilon_linear" else x[splits["validation"]]
        train_input = params[splits["train"], 0] if name == "epsilon_linear" else x[splits["train"]]
        prediction = decode(model.predict(val_input), sigma.shape[1:])
        validation[name] = float(np.sqrt(np.mean(np.abs(
            prediction - sigma[splits["validation"]]) ** 2)))
        delta = prediction - sigma[splits["validation"]]
        train_prediction = decode(model.predict(train_input), sigma.shape[1:])
        train_delta = train_prediction - sigma[splits["train"]]
        validation_details[name] = {
            "complex_rmse": validation[name],
            "train_complex_rmse": float(np.sqrt(np.mean(np.abs(train_delta) ** 2))),
            "train_first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(train_delta[:, :8]) ** 2))),
            "real_rmse": float(np.sqrt(np.mean(delta.real ** 2))),
            "imaginary_rmse": float(np.sqrt(np.mean(delta.imag ** 2))),
            "first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(delta[:, :8]) ** 2))),
            "last_8_complex_rmse": float(np.sqrt(np.mean(np.abs(delta[:, -8:]) ** 2))),
            "positive_imaginary_fraction": float(np.mean(prediction.imag > 1e-10)),
            "actual_latent_dimension": getattr(model, "actual_dimension", None),
            "optimizer_iterations": getattr(model.model, "n_iter_", None),
            "model_parameter_count": (
                sum(a.size for a in model.model.coefs_ + model.model.intercepts_)
                if hasattr(model.model, "coefs_") else None
            ),
        }
        trained[name] = model
    selected = min(validation, key=validation.get)
    metrics = {
        "dataset_sha256": sha256_file(dataset),
        "seed": seed, "split_seed": seed, "model_seed": model_seed,
        "python": platform.python_version(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "candidates": candidates,
        "split_rule": rule, "split_type": split, "input_includes_u": split == "u",
        "feature_order": "flattened G0 real, G0 imaginary" + (", U" if split == "u" else ""),
        "fit_warnings": fit_warnings,
        "validation_diagnostics": validation_details,
        "splits": splits, "validation_complex_rmse": validation, "selected": selected,
        "evaluation_phase": "validation_only" if validation_only else "exploratory_test",
        "note": "Exploratory finite-bath AIM; not lattice DMFT or a final test.",
    }
    if include_epsilon_linear:
        linear = trained["epsilon_linear"]
        metrics["epsilon_linear"] = {
            "input": "epsilon_d only; other candidates use G0",
            "policy": "piecewise linear interpolation; linear extrapolation using nearest endpoint pair",
            "duplicate_training_epsilon": "error",
            "training_range": [float(linear.epsilon[0]), float(linear.epsilon[-1])],
            "validation_coverage": linear.coverage(params[splits["validation"], 0]),
        }
    output.mkdir(parents=True)
    if save_candidates:
        bundle = output / "candidates.joblib"
        joblib.dump(trained, bundle)
        metrics["candidate_bundle_sha256"] = sha256_file(bundle)
        metrics["saved_candidates"] = list(trained)
    diagnostics = {
        "interpretation": "Oracle offsets use test targets and are descriptive only; "
                          "sign checks are necessary, not sufficient, for causality.",
        "network_training_and_bottleneck": network_diagnostics,
        "test_samples": [],
    }
    if not validation_only:
        test_input = params[splits["test"], 0] if selected == "epsilon_linear" else x[splits["test"]]
        prediction = decode(trained[selected].predict(test_input), sigma.shape[1:])
        if include_epsilon_linear:
            metrics["epsilon_linear"]["test_coverage"] = linear.coverage(params[splits["test"], 0])
        target = sigma[splits["test"]]
        difference = prediction - target
        metrics.update({
            "test_complex_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
            "test_first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(difference[:, :8]) ** 2))),
            "test_last_8_complex_rmse": float(np.sqrt(np.mean(np.abs(difference[:, -8:]) ** 2))),
            "test_real_rmse": float(np.sqrt(np.mean(difference.real ** 2))),
            "test_imaginary_rmse": float(np.sqrt(np.mean(difference.imag ** 2))),
            "test_positive_imaginary_fraction": float(np.mean(prediction.imag > 1e-10)),
            "reference_positive_imaginary_fraction": float(np.mean(target.imag > 1e-10)),
        })
        for position, index in enumerate(splits["test"]):
            err = difference[position]
            offset = float(err.real.mean())
            bad = np.flatnonzero(prediction[position].imag > 1e-10)
            diagnostics["test_samples"].append({
                "index": index, "epsilon_d": float(params[index, 0]),
                "complex_rmse": float(np.sqrt(np.mean(np.abs(err) ** 2))),
                "oracle_real_offset": offset,
                "oracle_centered_real_rmse": float(np.sqrt(np.mean((err.real - offset) ** 2))),
                "last_8_signed_real_error": float(err.real[-8:].mean()),
                "positive_imaginary_points": [
                    {"frequency": float(omega[j]), "prediction": float(prediction[position, j].imag),
                     "reference": float(target[position, j].imag)} for j in bad
                ],
            })
    if include_latent:
        pca_model = trained["pca_ridge_16"]
        pca = pca_model.pca
        scaler = pca_model.y_scaler
        pca_diagnostics = {}
        for dim in (1, 2, 4, 8, 16):
            part = {"explained_variance_ratio_sum": float(pca.explained_variance_ratio_[:dim].sum())}
            for role in ("train", "validation"):
                indices = splits[role]
                scaled = scaler.transform(y[indices])
                components = pca.components_[:dim]
                projected = (scaled - pca.mean_) @ components.T @ components + pca.mean_
                reconstructed = decode(scaler.inverse_transform(projected), sigma.shape[1:])
                part[f"{role}_oracle_reconstruction_rmse"] = float(
                    np.sqrt(np.mean(np.abs(reconstructed - sigma[indices]) ** 2))
                )
            pca_diagnostics[str(dim)] = part
        diagnostics["pca_train_fit_oracle_reconstruction"] = pca_diagnostics
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    joblib.dump(trained[selected], output / "selected_model.joblib")
    if not validation_only:
        np.savez_compressed(output / "test_predictions.npz", indices=np.asarray(splits["test"]),
                            parameters=params[splits["test"]], omega=omega,
                            reference=target, prediction=prediction)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, help="Model initialization seed (default: split seed)")
    parser.add_argument("--split", choices=("epsilon", "random", "u"), default="epsilon")
    parser.add_argument("--include-latent", action="store_true",
                        help="Compare PCA and unconstrained MLP bottlenecks 1, 2, 4, 8, 16")
    parser.add_argument("--compare-activations", action="store_true",
                        help="Also compare tanh with ReLU in all MLP hidden layers")
    parser.add_argument("--validation-only", action="store_true",
                        help="Do not predict or score the reserved test samples")
    parser.add_argument("--include-epsilon-linear", action="store_true",
                        help="Add epsilon_d interpolation with endpoint linear extrapolation (fixed bath/U only)")
    parser.add_argument("--save-candidates", action="store_true",
                        help="Save all fitted candidates for a frozen external comparison")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.seed, args.split,
                         args.include_latent, args.model_seed, args.compare_activations,
                         args.validation_only, args.include_epsilon_linear,
                         args.save_candidates), indent=2))


if __name__ == "__main__":
    main()

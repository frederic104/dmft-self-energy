"""Compare Sigma-only and Sigma+G refinement on an existing AIM study."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from .ed_validation import analyze, load_candidates
from .learning.dyson_objective import refine_model
from .learning.models import decode, encode
from .provenance import sha256_file


def run_comparison(study: Path, output: Path, *, candidates=("mlp", "latent_mlp_8"),
                   green_weight: float = 1., max_iter: int = 100) -> dict:
    """Refine independent copies of saved models; never train on held-out targets.

    Unmodified candidates remain in the comparison. Each new pair shares the
    same checkpoint, preprocessing, L2 penalty and L-BFGS-B iteration/evaluation
    caps. Only the G-loss weight differs. Actual optimizer termination is saved.
    """
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (not candidates or len(set(candidates)) != len(candidates)
            or not np.isfinite(green_weight) or green_weight <= 0 or max_iter < 1):
        raise ValueError("Need unique candidates, positive G weight and iteration budget")
    original = json.loads((study / "recipe.json").read_text())
    if original["evaluation_phase"] != "validation_only":
        raise ValueError("Expected a validation-only AIM study")
    recipe = {"source_study": str(study.resolve()), "source_recipe_sha256": sha256_file(study / "recipe.json"),
              "evaluation_phase": "validation_only", "candidates": list(candidates),
              "green_weight": green_weight, "max_iter": max_iter,
              "selection": "Sigma validation RMSE only; assess paired changes in G and occupation separately",
              "source_recipe": original, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    source = Path(__file__).parent
    recipe["source_sha256"] = {name: sha256_file(source / name) for name in (
        "ed_objective.py", "learning/dyson_objective.py", "learning/models.py",
        "ed_validation.py", "ed_properties.py")}
    output.mkdir(parents=True)
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    rows, failures = [], []
    for axis in original["axes"]:
        dataset = study / axis / "dataset.npz"
        for seed in original["model_seeds"]:
            source_run = study / axis / f"seed_{seed}"
            metrics, models = load_candidates(dataset, source_run)
            if metrics["input_includes_u"] or metrics["split_type"] != "bath":
                raise ValueError("This comparison uses the existing fixed-U bath study")
            train, validation, test = (metrics["splits"][k] for k in ("train", "validation", "test"))
            if (not train or not validation or set(train) & set(validation)
                    or set(train) & set(test) or set(validation) & set(test)):
                raise ValueError("Invalid or overlapping partitions")
            if any(name not in models for name in candidates):
                raise ValueError("Requested starting candidate was not saved")
            with np.load(dataset, allow_pickle=False) as data:
                # Only train targets enter refinement; test arrays are never scored.
                g0, sigma, green = (data[k][train].copy() for k in ("g0", "sigma", "green"))
                val_g0, val_sigma = (data[k][validation].copy() for k in ("g0", "sigma"))
            fit_info = {}
            for name in candidates:
                for treatment, weight in (("sigma", 0.), ("sigma_g", green_weight)):
                    label = f"{name}_refine_{treatment}"
                    print(f"Running {axis}, seed {seed}, {label}", flush=True)
                    try:
                        fitted, info = refine_model(models[name], encode(g0), sigma, g0, green,
                                                    weight=weight, max_iter=max_iter)
                        prediction = decode(fitted.predict(encode(val_g0)), val_sigma.shape[1:])
                        if not np.isfinite(prediction).all():
                            raise ValueError("Non-finite validation prediction")
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        failure = {"axis": axis, "model_seed": seed, "candidate": label,
                                   "status": "failed", "reason": str(exc)}
                        failures.append(failure)
                        fit_info[label] = failure
                        continue
                    models[label] = fitted
                    fit_info[label] = info
                    delta = prediction - val_sigma
                    rmse = float(np.sqrt(np.mean(np.abs(delta)**2)))
                    metrics["validation_complex_rmse"][label] = rmse
                    metrics["validation_diagnostics"][label] = {
                        "complex_rmse": rmse,
                        "first_8_complex_rmse": float(np.sqrt(np.mean(np.abs(delta[:, :8])**2))),
                        "positive_imaginary_fraction": float(np.mean(prediction.imag > 1e-10))}
                    metrics["fit_warnings"][label] = [] if info["success"] else [info["message"]]
                    metrics["candidates"].append({**fitted.specification, "label": label,
                                                  "refinement_weight": weight})
            result = output / axis / f"seed_{seed}"
            result.mkdir(parents=True)
            metrics["source_metrics_sha256"] = sha256_file(source_run / "metrics.json")
            metrics["source_candidate_bundle_sha256"] = metrics["candidate_bundle_sha256"]
            metrics["refinement"] = fit_info
            # Keep original training versions labeled as such; record current versions explicitly.
            metrics["refinement_versions"] = {k: recipe[k] for k in ("python", "numpy", "scipy", "sklearn")}
            metrics["selected"] = min(metrics["validation_complex_rmse"], key=metrics["validation_complex_rmse"].get)
            joblib.dump(models, result / "candidates.joblib")
            joblib.dump(models[metrics["selected"]], result / "selected_model.joblib")
            metrics["candidate_bundle_sha256"] = sha256_file(result / "candidates.joblib")
            metrics["saved_candidates"] = list(models)
            (result / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            diagnostics = analyze(dataset, result, result / "analysis", properties=True)
            for name, scores in metrics["validation_diagnostics"].items():
                prop = diagnostics["properties"][name]
                info = fit_info.get(name, {})
                rows.append({"axis": axis, "model_seed": seed, "candidate": name,
                             "sigma_rmse": scores["complex_rmse"],
                             "sigma_first_8_rmse": scores["first_8_complex_rmse"],
                             "sigma_positive_imaginary_fraction": scores["positive_imaginary_fraction"],
                             "selected_by_sigma": name == metrics["selected"],
                             "optimizer_success": info.get("success"),
                             "optimizer_iterations": info.get("iterations"),
                             "property_status": prop["status"],
                             "property_failure": prop.get("reason", ""),
                             **{key: prop.get(key) for key in (
                                 "green_complex_rmse", "green_first_8_complex_rmse",
                                 "occupancy_tail_vs_exact_rmse", "reference_tail_vs_exact_rmse",
                                 "occupancy_tail_out_of_bounds_fraction")}})
            with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    summary = {"recipe": recipe, "comparisons": rows, "failures": failures,
               "interpretation": "Matched checkpoint refinement, not from-scratch training or a causal decoder. Validation only; no property target used as an input."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["mlp", "latent_mlp_8"])
    parser.add_argument("--green-weight", type=float, default=1.)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    run_comparison(args.study, args.output, candidates=tuple(args.candidates),
                   green_weight=args.green_weight, max_iter=args.max_iter)
    print(f"Saved {args.output / 'comparison.csv'}")


if __name__ == "__main__":
    main()

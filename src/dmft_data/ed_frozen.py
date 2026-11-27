"""Freeze three AIM candidates, then evaluate unseen epsilon states without fitting."""

import argparse
import csv
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from .ed import generate_small_dataset
from .ed_properties import compare_properties
from .ed_validation import load_candidates
from .learning.models import decode, encode
from .provenance import sha256_file

PANEL = ("pca_ridge_8", "grid_latent_8_refine_sigma_g", "causal_latent_8_refine_sigma")


def freeze(study: Path, result: Path, output: Path, *, evaluation_seed=441, samples=40):
    """Copy a fixed panel from the uniform decoder experiment; no fitting/selection."""
    study, result, output = map(Path, (study, result, output))
    if output.exists():
        raise FileExistsError(output)
    recipe = json.loads((study / "recipe.json").read_text())
    if samples < 1 or evaluation_seed == recipe["dataset_seed"]:
        raise ValueError("Need positive samples and a new evaluation seed")
    protocol = {"version": 1, "panel": list(PANEL), "source_recipe": recipe,
                "evaluation_seed": evaluation_seed, "samples": samples,
                "evaluation_scale": recipe["validation_scale"],
                "scope": "New epsilon draws at the development validation bath scale; reserved test scale unopened",
                "decision": "Report Sigma, first-eight G and per-spin occupation errors; Pareto comparison to PCA8, no automatic winner",
                "training_policy": "Existing PCA8/Ridge alpha=1; reset latent8 grid+G weight=1 and uniform16 causal Sigma-only; L-BFGS-B cap=100, alpha=.001",
                "cases": []}
    bundles = {}
    for axis in recipe["axes"]:
        dataset = study / axis / "dataset.npz"
        with np.load(dataset) as data:
            epsilon = data["parameters"][:, 0].copy()
        for seed in recipe["model_seeds"]:
            run = result / axis / f"seed_{seed}"
            metrics, models = load_candidates(dataset, run)
            design = metrics.get("decoder_design", {})
            if (design.get("pole_count") != 16 or design.get("pole_spacing", "uniform") != "uniform"
                    or not np.allclose(design["poles"], np.linspace(-4, 4, 16))):
                raise ValueError("Freeze requires the uniform 16-pole [-4,4] decoder comparison")
            if any(name not in models for name in PANEL):
                raise ValueError("Missing fixed panel candidate")
            for name, weight in zip(PANEL[1:], (1., 0.)):
                info = metrics["refinement"][name]
                if info["weight"] != weight or info["max_iter"] != 100 or info["alpha"] != .001:
                    raise ValueError("Source training settings do not match the frozen policy")
            training_epsilon = epsilon[metrics["splits"]["train"]]
            key = f"{axis}_seed_{seed}"
            bundles[key] = {name: models[name] for name in PANEL}
            protocol["cases"].append({"key": key, "axis": axis, "model_seed": seed,
                                      "source_dataset_sha256": metrics["dataset_sha256"],
                                      "source_bundle_sha256": metrics["candidate_bundle_sha256"],
                                      "source_metrics_sha256": sha256_file(run / "metrics.json"),
                                      "seen_epsilon": np.unique(epsilon).tolist(),
                                      "training_epsilon_range": [float(training_epsilon.min()), float(training_epsilon.max())],
                                      "training_termination": {n: metrics["refinement"][n] for n in PANEL[1:]}})
    output.mkdir(parents=True)
    joblib.dump(bundles, output / "panel.joblib")
    protocol["bundle_sha256"] = sha256_file(output / "panel.joblib")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def evaluate(panel: Path, output: Path):
    """Use the predeclared seed/scale; no refitting, normalization update or reranking."""
    panel, output = Path(panel), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((panel / "protocol.json").read_text())
    if sha256_file(panel / "panel.joblib") != protocol["bundle_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    # Trusted local pickle artifacts only, as for existing candidate analysis.
    bundles = joblib.load(panel / "panel.joblib")
    recipe = protocol["source_recipe"]
    # Check proposed epsilon draws before any ED solves or predictions.
    proposed = np.random.default_rng(protocol["evaluation_seed"]).uniform(-1., .2, protocol["samples"])
    for case in protocol["cases"]:
        if np.isclose(proposed[:, None], case["seen_epsilon"], atol=1e-12, rtol=0).any():
            raise ValueError("Evaluation epsilon overlaps development states")
    output.mkdir(parents=True)
    datasets, rows, coverage = {}, [], {}
    for axis in recipe["axes"]:
        path = output / axis / "dataset.npz"
        generate_small_dataset(path, samples=protocol["samples"], seed=protocol["evaluation_seed"],
                               beta=recipe["beta"], frequency_count=recipe["frequency_count"],
                               fixed_u=recipe["fixed_u"], fixed_bath=tuple(recipe["base_bath"]),
                               bath_scales=(protocol["evaluation_scale"],), bath_axis=axis)
        datasets[axis] = sha256_file(path)
    for case in protocol["cases"]:
        axis, seed, key = case["axis"], case["model_seed"], case["key"]
        with np.load(output / axis / "dataset.npz") as data:
            g0, sigma, green, occupation, omega, params = (data[k].copy() for k in (
                "g0", "sigma", "green", "occupancy_per_spin", "omega", "parameters"))
        lo, hi = case["training_epsilon_range"]
        coverage[key] = {"below": int(np.sum(params[:, 0] < lo)), "above": int(np.sum(params[:, 0] > hi)),
                         "within": int(np.sum((params[:, 0] >= lo) & (params[:, 0] <= hi)))}
        arrays = {"parameters": params, "omega": omega, "reference_sigma": sigma,
                  "reference_green": green, "reference_occupancy_per_spin": occupation}
        case_rows = []
        for name in PANEL:
            row = {"axis": axis, "model_seed": seed, "candidate": name}
            try:
                predicted = decode(bundles[key][name].predict(encode(g0)), sigma.shape[1:])
                prop, values = compare_properties(g0, predicted, green, occupation, omega, recipe["beta"])
                error = predicted - sigma
                row.update(sigma_rmse=float(np.sqrt(np.mean(np.abs(error)**2))),
                           sigma_first_8_rmse=float(np.sqrt(np.mean(np.abs(error[:, :8])**2))),
                           sigma_positive_imaginary_fraction=float(np.mean(predicted.imag > 1e-10)), **prop)
                arrays[f"{name}_sigma"] = predicted
                arrays.update({f"{name}_{k}": v for k, v in values.items()})
            except (ValueError, FloatingPointError) as exc:
                row.update(status="failed", reason=str(exc))
            case_rows.append(row)
        baseline = case_rows[0]
        keys = ("sigma_rmse", "green_first_8_complex_rmse", "occupancy_tail_vs_exact_rmse")
        for row in case_rows:
            row["dominates_pca8"] = (all(row[k] <= baseline[k] for k in keys)
                                     and any(row[k] < baseline[k] for k in keys)
                                     if row["status"] == baseline["status"] == "ok" else None)
        rows.extend(case_rows)
        np.savez_compressed(output / axis / f"predictions_seed_{seed}.npz", **arrays)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"phase": "frozen_unseen_epsilon", "protocol_sha256": sha256_file(panel / "protocol.json"),
               "model_bundle_sha256": protocol["bundle_sha256"], "dataset_sha256": datasets,
               "protocol": protocol, "epsilon_coverage": coverage, "comparisons": rows,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "source_sha256": {n: sha256_file(Path(__file__).parent / n) for n in (
                   "ed_frozen.py", "ed.py", "ed_properties.py", "learning/models.py", "learning/causal_decoder.py")},
               "interpretation": "Conditional on development selection; not material/extrapolation evidence. Reusing these scores to change models makes this development data."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("freeze")
    for name in ("study", "result", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--evaluation-seed", type=int, default=441)
    prepare.add_argument("--samples", type=int, default=40)
    assess = sub.add_parser("evaluate")
    for name in ("panel", "output"):
        assess.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.study, args.result, args.output, evaluation_seed=args.evaluation_seed, samples=args.samples)
    else:
        evaluate(args.panel, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Freeze three AIM candidates, then evaluate unseen epsilon states without fitting."""

import argparse
import csv
import json
import platform
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from .ed import generate_small_dataset
from .ed_properties import compare_properties
from .ed_validation import load_candidates
from .learning.causal_decoder import decoder_starts
from .learning.dyson_objective import refine_model
from .learning.models import decode, encode
from .provenance import sha256_file

PANEL = ("pca_ridge_8", "grid_latent_8_refine_sigma_g", "causal_latent_8_refine_sigma")
FINAL_SEEDS = (31, 32, 33)


def prepare_final(panel: Path, study: Path, result: Path, output: Path):
    """Close search: preserve seed31, train only the two finalists at seeds32/33.

    Copies reserved source states for later explicit evaluation; never scores
    them or uses their targets in optimization. No tunable training arguments.
    """
    panel, study, result, output = map(Path, (panel, study, result, output))
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((panel / "protocol.json").read_text())
    if (protocol["panel"] != list(PANEL) or sha256_file(panel / "panel.joblib") != protocol["bundle_sha256"]
            or any(c["model_seed"] != 31 for c in protocol["cases"])):
        raise ValueError("Expected the unchanged seed31 frozen panel")
    bundles = joblib.load(panel / "panel.joblib")
    original_cases = list(protocol["cases"])
    protocol.update(version=2, evaluation_scope="reserved_scale", search_status="closed",
                    model_seeds=list(FINAL_SEEDS), parent_protocol_sha256=sha256_file(panel / "protocol.json"),
                    parent_bundle_sha256=protocol["bundle_sha256"],
                    evaluation_scale=protocol["source_recipe"]["reserved_test_scale"],
                    scope="Original reserved bath scale; epsilon shared with training, no unseen-epsilon claim",
                    preparation_versions={"python": platform.python_version(), "numpy": np.__version__,
                                          "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                    preparation_source_sha256={n: sha256_file(Path(__file__).parent / n) for n in (
                        "ed_frozen.py", "learning/dyson_objective.py", "learning/causal_decoder.py")},
                    reserved_datasets={})
    protocol.pop("evaluation_seed", None)
    protocol.pop("samples", None)
    # Output path also preserves a failed preparation record without overwriting prior panels.
    output.mkdir(parents=True)
    for base in original_cases:
        axis = base["axis"]
        dataset = study / axis / "dataset.npz"
        source_run = result / axis / "seed_31"
        metrics, models = load_candidates(dataset, source_run)
        if (metrics["dataset_sha256"] != base["source_dataset_sha256"]
                or metrics["candidate_bundle_sha256"] != base["source_bundle_sha256"]
                or sha256_file(source_run / "metrics.json") != base["source_metrics_sha256"]):
            raise ValueError("Preparation source differs from the frozen panel")
        train, test = (metrics["splits"][k] for k in ("train", "test"))
        if not test or set(train) & set(test) or set(metrics["splits"]["validation"]) & set(test):
            raise ValueError("Invalid reserved partition")
        with np.load(dataset) as data:
            g0, sigma, green = (data[k][train].copy() for k in ("g0", "sigma", "green"))
            omega = data["omega"].copy()
            reserved = {k: data[k][test].copy() for k in (
                "g0", "sigma", "green", "occupancy_per_spin", "parameters", "bath_scale")}
        if not np.all(reserved["bath_scale"] == protocol["evaluation_scale"]):
            raise ValueError("Reserved bath scale does not match the protocol")
        reserved.update(omega=omega, source_indices=np.array(test))
        target = output / axis
        target.mkdir()
        np.savez_compressed(target / "reserved.npz", **reserved)
        protocol["reserved_datasets"][axis] = sha256_file(target / "reserved.npz")
        for seed in FINAL_SEEDS[1:]:
            starts, _ = decoder_starts(models["latent_mlp_8"], sigma, omega, seed=seed)
            case = deepcopy(base)
            case.update(key=f"{axis}_seed_{seed}", model_seed=seed, training_termination={})
            selected = {PANEL[0]: bundles[base["key"]][PANEL[0]]}
            for name, start_key, weight in zip(PANEL[1:], ("grid_latent_8", "causal_latent_8"), (1., 0.)):
                print(f"Preparing {axis}, seed {seed}, {name}", flush=True)
                try:
                    selected[name], info = refine_model(starts[start_key], encode(g0), sigma, g0, green,
                                                       weight=weight, max_iter=100, alpha=.001)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    (output / "preparation_failure.json").write_text(json.dumps(
                        {"axis": axis, "seed": seed, "candidate": name, "reason": str(exc)}, indent=2))
                    raise
                case["training_termination"][name] = info
            bundles[case["key"]] = selected
            protocol["cases"].append(case)
    joblib.dump(bundles, output / "panel.joblib")
    protocol["bundle_sha256"] = sha256_file(output / "panel.joblib")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def seed_summary(rows):
    """Optimizer variability on shared states; not a physical-sample confidence interval."""
    result = []
    for axis in dict.fromkeys(r["axis"] for r in rows):
        for name in PANEL:
            group = [r for r in rows if r["axis"] == axis and r["candidate"] == name]
            good = [r for r in group if r["status"] == "ok"]
            item = {"axis": axis, "candidate": name, "seed_count": len(group), "successful_seeds": len(good)}
            for key in ("sigma_rmse", "green_first_8_complex_rmse", "occupancy_tail_vs_exact_rmse"):
                values = [r[key] for r in good]
                item[key + "_mean"] = float(np.mean(values)) if values else None
                item[key + "_std"] = float(np.std(values)) if len(values) > 1 else None
            result.append(item)
    return result


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
    reserved_mode = protocol.get("evaluation_scope") == "reserved_scale"
    # Check proposed epsilon draws before any ED solves or predictions.
    proposed = (None if reserved_mode else np.random.default_rng(protocol["evaluation_seed"]).uniform(
        -1., .2, protocol["samples"]))
    for case in ([] if reserved_mode else protocol["cases"]):
        if np.isclose(proposed[:, None], case["seen_epsilon"], atol=1e-12, rtol=0).any():
            raise ValueError("Evaluation epsilon overlaps development states")
    output.mkdir(parents=True)
    datasets, rows, coverage = {}, [], {}
    for axis in recipe["axes"]:
        path = output / axis / "dataset.npz"
        if reserved_mode:
            source = panel / axis / "reserved.npz"
            if sha256_file(source) != protocol["reserved_datasets"][axis]:
                raise ValueError("Reserved dataset checksum mismatch")
            path.parent.mkdir(parents=True)
            path.write_bytes(source.read_bytes())
        else:
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
    summary = {"phase": "frozen_reserved_scale" if reserved_mode else "frozen_unseen_epsilon", "protocol_sha256": sha256_file(panel / "protocol.json"),
               "model_bundle_sha256": protocol["bundle_sha256"], "dataset_sha256": datasets,
               "protocol": protocol, "epsilon_coverage": coverage, "comparisons": rows,
               "seed_summary": seed_summary(rows),
               "seed_summary_note": "Population SD across initialization seeds on shared states; PCA is reused, not independent retraining. No automatic winner.",
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "source_sha256": {n: sha256_file(Path(__file__).parent / n) for n in (
                   "ed_frozen.py", "ed.py", "ed_properties.py", "learning/models.py", "learning/causal_decoder.py")},
               "interpretation": "Conditional on development selection; not material/extrapolation evidence. Reusing these scores to change models makes this development data."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["seed_summary"][0]))
        writer.writeheader()
        writer.writerows(summary["seed_summary"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("freeze")
    for name in ("study", "result", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--evaluation-seed", type=int, default=441)
    prepare.add_argument("--samples", type=int, default=40)
    final = sub.add_parser("prepare-final")
    for name in ("panel", "study", "result", "output"):
        final.add_argument(f"--{name}", type=Path, required=True)
    assess = sub.add_parser("evaluate")
    for name in ("panel", "output"):
        assess.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.study, args.result, args.output, evaluation_seed=args.evaluation_seed, samples=args.samples)
    elif args.command == "prepare-final":
        prepare_final(args.panel, args.study, args.result, args.output)
    else:
        evaluate(args.panel, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

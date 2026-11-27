"""Run a small AIM representation-to-property study with one command."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .ed import generate_small_dataset
from .ed_baselines import run
from .ed_validation import analyze
from .provenance import sha256_file


def run_study(output: Path, *, samples: int = 40, seed: int = 31,
              model_seeds: tuple[int, ...] = (31,), axes: tuple[str, ...] = ("coupling", "energy"),
              scales: tuple[float, ...] = (.7, .85, 1., 1.15, 1.3),
              validation_scale: float = .85, test_scale: float = 1.15,
              fixed_u: float = 2., beta: float = 20., frequency_count: int = 64,
              base_bath: tuple[float, ...] = (-1., 1., .5, .7)) -> dict:
    """Generate paired cases, fit existing baselines and propagate to properties.

    All scored outputs use validation states. Each axis is a separate matched
    experiment, not a combined two-dimensional bath grid. No independent test is
    consumed and no properties are supplied as predictor inputs.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (not axes or len(set(axes)) != len(axes) or not set(axes) <= {"coupling", "energy"}
            or not model_seeds or len(set(model_seeds)) != len(model_seeds)):
        raise ValueError("Specify unique valid axes and model seeds")
    if (validation_scale not in scales or test_scale not in scales or validation_scale == test_scale
            or (len(scales) - 2) * samples <= 16 or frequency_count < 16):
        raise ValueError("Need distinct holdouts, at least 17 training states and 16 frequencies")
    recipe = {"samples_per_bath": samples, "dataset_seed": seed, "model_seeds": model_seeds,
              "axes": axes, "scales": scales, "validation_scale": validation_scale,
              "reserved_test_scale": test_scale, "fixed_u": fixed_u, "beta": beta,
              "frequency_count": frequency_count, "base_bath": base_bath,
              "evaluation_phase": "validation_only", "property_method": "Dyson G; two-term fitted-tail occupancy",
              "selection": "validation Sigma complex RMSE; property errors are separate diagnostics"}
    source = Path(__file__).parent
    recipe["source_sha256"] = {name: sha256_file(source / name) for name in (
        "ed.py", "ed_baselines.py", "ed_validation.py", "ed_properties.py", "ed_workflow.py",
        "learning/models.py", "learning/splits.py")}
    output.mkdir(parents=True)
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    rows = []
    for axis in axes:
        case = output / axis
        dataset = case / "dataset.npz"
        generate_small_dataset(dataset, samples=samples, seed=seed, fixed_u=fixed_u,
                               beta=beta, frequency_count=frequency_count,
                               fixed_bath=base_bath, bath_scales=scales, bath_axis=axis)
        for model_seed in model_seeds:
            result = case / f"seed_{model_seed}"
            print(f"Running {axis}, model seed {model_seed}", flush=True)
            metrics = run(dataset, result, seed=seed, split="bath", include_latent=True,
                          model_seed=model_seed, validation_only=True, save_candidates=True,
                          include_ridge=True, validation_bath_scale=validation_scale,
                          test_bath_scale=test_scale)
            diagnostics = analyze(dataset, result, result / "analysis", properties=True)
            for candidate, scores in metrics["validation_diagnostics"].items():
                prop = diagnostics["properties"][candidate]
                rows.append({"axis": axis, "model_seed": model_seed, "candidate": candidate,
                             "sigma_rmse": scores["complex_rmse"],
                             "sigma_first_8_rmse": scores["first_8_complex_rmse"],
                             "sigma_positive_imaginary_fraction": scores["positive_imaginary_fraction"],
                             "selected_by_sigma": candidate == metrics["selected"],
                             "property_status": prop["status"],
                             "property_failure": prop.get("reason", ""),
                             **{key: prop.get(key) for key in (
                                 "green_complex_rmse", "green_first_8_complex_rmse",
                                 "occupancy_tail_vs_exact_rmse", "reference_tail_vs_exact_rmse",
                                 "occupancy_tail_model_vs_reference_method_rmse",
                                 "occupancy_tail_out_of_bounds_fraction")}})
            # Preserve completed cases even if a later run fails.
            with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {"recipe": recipe, "comparisons": rows,
               "interpretation": "Exploratory paired AIM states. Fitted-tail occupancy is approximate; compare its ED-reference error. No lattice DOS or real-frequency prediction."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[31])
    parser.add_argument("--axes", choices=("coupling", "energy"), nargs="+", default=["coupling", "energy"])
    parser.add_argument("--scales", type=float, nargs="+", default=[.7, .85, 1., 1.15, 1.3])
    parser.add_argument("--validation-scale", type=float, default=.85)
    parser.add_argument("--test-scale", type=float, default=1.15)
    parser.add_argument("--fixed-u", type=float, default=2.)
    parser.add_argument("--beta", type=float, default=20.)
    parser.add_argument("--frequency-count", type=int, default=64)
    parser.add_argument("--base-bath", type=float, nargs=4, default=[-1., 1., .5, .7])
    args = parser.parse_args()
    run_study(args.output, samples=args.samples, seed=args.seed, model_seeds=tuple(args.model_seeds),
              axes=tuple(args.axes), scales=tuple(args.scales), validation_scale=args.validation_scale,
              test_scale=args.test_scale, fixed_u=args.fixed_u, beta=args.beta,
              frequency_count=args.frequency_count, base_bath=tuple(args.base_bath))
    print(f"Saved {args.output / 'comparison.csv'}")


if __name__ == "__main__":
    main()

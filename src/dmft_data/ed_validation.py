"""Frequency-resolved comparison of saved AIM validation candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from .learning.models import EpsilonLinear, decode, encode
from .provenance import sha256_file


def analyze(dataset: Path, result: Path, output: Path) -> dict:
    """Compare saved models on validation states, without fitting or scoring test."""
    dataset, result, output = map(Path, (dataset, result, output))
    if output.exists():
        raise FileExistsError(output)
    metrics_path, bundle = result / "metrics.json", result / "candidates.joblib"
    metrics = json.loads(metrics_path.read_text())
    if metrics["evaluation_phase"] != "validation_only":
        raise ValueError("Expected a validation-only result")
    if sha256_file(dataset) != metrics["dataset_sha256"]:
        raise ValueError("Dataset does not match validation run")
    if not metrics.get("candidate_bundle_sha256") or sha256_file(bundle) != metrics["candidate_bundle_sha256"]:
        raise ValueError("Candidate bundle does not match validation run")
    # Joblib loads pickle; use only artifacts from a trusted local run.
    models = joblib.load(bundle)
    if not isinstance(models, dict) or list(models) != metrics["saved_candidates"]:
        raise ValueError("Candidate names do not match validation run")
    with np.load(dataset, allow_pickle=False) as data:
        indices = np.asarray(metrics["splits"]["validation"], dtype=int)
        omega = data["omega"].copy()
        params = data["parameters"][indices].copy()
        g0, reference = data["g0"][indices].copy(), data["sigma"][indices].copy()
    if not len(indices) or not np.isfinite(reference).all():
        raise ValueError("Invalid validation states")
    rows, curves, summary = [], {}, {}
    for name, model in models.items():
        if (name == "epsilon_linear") != isinstance(model, EpsilonLinear):
            raise ValueError("Candidate type does not match name")
        inputs = params[:, 0] if name == "epsilon_linear" else encode(g0)
        if metrics["input_includes_u"] and name != "epsilon_linear":
            inputs = np.column_stack((inputs, params[:, 1]))
        prediction = decode(model.predict(inputs), reference.shape[1:])
        error = prediction - reference
        complex_rmse = np.sqrt(np.mean(np.abs(error) ** 2, axis=0))
        real_rmse = np.sqrt(np.mean(error.real ** 2, axis=0))
        imag_rmse = np.sqrt(np.mean(error.imag ** 2, axis=0))
        curves[name] = complex_rmse
        summary[name] = float(np.sqrt(np.mean(np.abs(error) ** 2)))
        if not np.isclose(summary[name], metrics["validation_complex_rmse"][name], rtol=1e-8, atol=1e-12):
            raise ValueError(f"Stored validation score does not match model: {name}")
        for i, w in enumerate(omega):
            rows.append((name, i, float(w), float(complex_rmse[i]), float(real_rmse[i]),
                         float(imag_rmse[i]), float(np.mean(error.real[:, i])),
                         float(np.mean(error.imag[:, i]))))
    output.mkdir(parents=True)
    with (output / "frequency_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("candidate", "frequency_index", "omega", "complex_rmse", "real_rmse",
                         "imag_rmse", "mean_signed_real_error", "mean_signed_imag_error"))
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, curve in curves.items():
        ax.plot(omega, curve, label=name)
    ax.set(xscale="log", xlabel="Matsubara frequency", ylabel="Complex RMSE",
           title="Validation states: frequency-resolved error")
    ax.legend(fontsize="x-small", ncol=2)
    fig.savefig(output / "frequency_errors.png", dpi=150)
    plt.close(fig)
    report = {"phase": "validation_only", "validation_count": len(indices),
              "split_rule": metrics["split_rule"], "dataset_sha256": metrics["dataset_sha256"],
              "candidate_bundle_sha256": metrics["candidate_bundle_sha256"],
              "complex_rmse": summary,
              "note": "epsilon_linear uses epsilon_d; other candidates use G0 (and U if enabled)."}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "result", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.result, args.output), indent=2))


if __name__ == "__main__":
    main()

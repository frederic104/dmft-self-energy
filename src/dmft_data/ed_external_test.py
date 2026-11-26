"""Evaluate a frozen validation winner on separate fixed-bath AIM states."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .learning.models import EpsilonLinear, decode, encode
from .provenance import sha256_file


def evaluate(development: Path, result: Path, test: Path, output: Path,
             all_candidates: bool = False) -> dict:
    """Score frozen model(s); no fitting or selection occurs here."""
    development, result, test, output = map(Path, (development, result, test, output))
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    metrics_path = result / "metrics.json"
    model_path = result / ("candidates.joblib" if all_candidates else "selected_model.joblib")
    metrics = json.loads(metrics_path.read_text())
    if metrics["evaluation_phase"] != "validation_only":
        raise ValueError("Result must come from a validation-only run")
    if sha256_file(development) != metrics["dataset_sha256"]:
        raise ValueError("Development dataset does not match validation result")
    if metrics["split_type"] not in ("random", "epsilon"):
        raise ValueError("Only fixed-U, fixed-bath tasks are supported")
    with np.load(development, allow_pickle=False) as d, np.load(test, allow_pickle=False) as t:
        dp, tp = d["parameters"].copy(), t["parameters"].copy()
        dw, tw = d["omega"].copy(), t["omega"].copy()
        g0, sigma = t["g0"].copy(), t["sigma"].copy()
        expected = ["epsilon_d", "interaction_u", "epsilon_b0", "epsilon_b1", "V_b0",
                    "V_b1", "chemical_potential", "beta"]
        if d["parameter_columns"].tolist() != expected or t["parameter_columns"].tolist() != expected:
            raise ValueError("Unexpected parameter schema")
    if (dp.ndim != 2 or tp.ndim != 2 or dp.shape[1] != 8 or tp.shape[1] != 8
            or not len(dp) or not len(tp) or dw.ndim != 1 or not len(dw)
            or g0.shape != (len(tp), len(tw)) or sigma.shape != g0.shape
            or not all(np.isfinite(a).all() for a in (dp, tp, dw, tw, g0, sigma))):
        raise ValueError("Invalid arrays")
    if not np.array_equal(dw, tw):
        raise ValueError("Matsubara meshes differ")
    if not (np.all(dp[:, 1:] == dp[0, 1:]) and np.all(tp[:, 1:] == dp[0, 1:])):
        raise ValueError("U, bath, mu and beta must match and be fixed")
    if len(np.unique(tp[:, 0])) != len(tp) or np.isin(tp[:, 0], dp[:, 0]).any():
        raise ValueError("Test states must be unique and distinct from all development states")
    # Joblib uses pickle; only load artifacts created by a trusted local run.
    selected = metrics["selected"]
    if all_candidates and (not metrics.get("candidate_bundle_sha256") or
                           sha256_file(model_path) != metrics["candidate_bundle_sha256"]):
        raise ValueError("Candidate bundle missing or does not match validation metrics")
    loaded = joblib.load(model_path)
    models = loaded if all_candidates else {selected: loaded}
    if all_candidates and (not isinstance(models, dict) or
                           list(models) != metrics.get("saved_candidates") or
                           set(models) != set(metrics["validation_complex_rmse"])):
        raise ValueError("Candidate bundle does not match validation candidates")
    train_eps = dp[metrics["splits"]["train"], 0]
    masks = {"below": tp[:, 0] < train_eps.min(),
             "within_inclusive": (tp[:, 0] >= train_eps.min()) & (tp[:, 0] <= train_eps.max()),
             "above": tp[:, 0] > train_eps.max()}
    candidate_groups, predictions = {}, {}
    for candidate, model in models.items():
        if (candidate == "epsilon_linear") != isinstance(model, EpsilonLinear):
            raise ValueError("Model type does not match candidate")
        inputs = tp[:, 0] if candidate == "epsilon_linear" else encode(g0)
        prediction = decode(model.predict(inputs), sigma.shape[1:])
        predictions[candidate] = prediction
        groups = {}
        for name, mask in masks.items():
            entry = {"count": int(mask.sum())}
            if mask.any():
                difference = prediction[mask] - sigma[mask]
                entry.update(complex_rmse=float(np.sqrt(np.mean(np.abs(difference) ** 2))),
                             first_8_complex_rmse=float(np.sqrt(np.mean(np.abs(difference[:, :8]) ** 2))),
                             real_rmse=float(np.sqrt(np.mean(difference.real ** 2))),
                             imaginary_rmse=float(np.sqrt(np.mean(difference.imag ** 2))),
                             positive_imaginary_fraction=float(np.mean(prediction[mask].imag > 1e-10)))
            groups[name] = entry
        candidate_groups[candidate] = groups
    report = {"selected": selected, "development_sha256": sha256_file(development),
              "test_sha256": sha256_file(test), "model_sha256": sha256_file(model_path),
              "validation_metrics_sha256": sha256_file(metrics_path),
              "training_epsilon_range": [float(train_eps.min()), float(train_eps.max())],
              "groups": candidate_groups[selected], "candidate_groups": candidate_groups,
              "python": platform.python_version(),
              "numpy": np.__version__, "sklearn": sklearn.__version__,
              "note": "Separate fixed-bath impurity states; sign check does not prove causality."}
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", parameters=tp, omega=tw,
                        reference=sigma, prediction=predictions[selected],
                        **{f"candidate_{name}": value for name, value in predictions.items()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "result", "test", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.development, args.result, args.test, args.output,
                              args.all_candidates), indent=2))


if __name__ == "__main__":
    main()

"""Plot a prepared Track A run against an existing trusted model prediction."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .learning.data import load_json
from .learning.models import decode, encode
from .plotting import prediction_figure
from .provenance import sha256_file


def visualize(prepared: Path, experiment: Path, source: str, raw_file: Path,
              output: Path, seed: int = 0) -> dict:
    """Predict one prepared run; identify its split and never call it a new test.

    Only load trusted joblib files: deserialization can execute arbitrary code.
    """
    prepared, experiment, output = map(Path, (prepared, experiment, output))
    if output.exists():
        raise FileExistsError(output)
    manifest_path = prepared / "manifest.json"
    arrays_path = prepared / "arrays.npz"
    manifest, metrics = load_json(manifest_path), load_json(experiment / "metrics.json")
    if (sha256_file(manifest_path) != metrics["prepared_manifest_sha256"]
            or sha256_file(arrays_path) != metrics["prepared_arrays_sha256"]
            or sha256_file(arrays_path) != manifest["arrays_sha256"]):
        raise ValueError("Prepared artifacts do not match the experiment")
    matches = [i for i, r in enumerate(manifest["records"])
               if r["relative_source_path"] == source]
    if len(matches) != 1:
        raise ValueError("Source must match exactly one prepared record")
    idx = matches[0]
    record = manifest["records"][idx]
    if sha256_file(raw_file) != record["source_sha256"]:
        raise ValueError("Raw file checksum does not match the prepared record")
    splits = load_json(experiment / "splits.json")
    roles = [role for role in ("train", "validation", "test")
             if idx in splits[role]["indices"]]
    if len(roles) != 1:
        raise ValueError("Record must belong to exactly one recorded split")
    layout = load_json(experiment / "feature_layout.json")
    if layout["input_blocks"] != manifest["input_blocks"] or layout["target_blocks"] != manifest["target_blocks"]:
        raise ValueError("Block layout differs from the prepared data")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        inputs, targets, meshes = (arrays[k].copy() for k in ("inputs", "targets", "meshes"))
    if (list(inputs.shape[1:]) != layout["input_shape"]
            or list(targets.shape[1:]) != layout["target_shape"]
            or len(inputs) != len(manifest["records"])
            or len(targets) != len(inputs) or len(meshes) != len(inputs)):
        raise ValueError("Prepared arrays do not match the recorded layout")
    if not all(np.isfinite(a).all() for a in (inputs[idx], targets[idx], meshes[idx])):
        raise ValueError("Non-finite input, reference or mesh")
    mesh = meshes[idx]
    if (mesh.ndim != 1 or np.any(mesh.imag <= 0) or np.any(np.diff(mesh.imag) <= 0)
            or not np.allclose(mesh.real, 0)):
        raise ValueError("Expected ordered positive Matsubara frequencies")
    features = np.concatenate([encode(inputs[idx:idx+1]), mesh.imag[None, :]], axis=1)
    keys = layout["parameter_features"]
    if keys:
        extra = np.asarray([[record[k] for k in keys]], dtype=float)
        if not np.isfinite(extra).all():
            raise ValueError("Non-finite parameter features")
        features = np.concatenate([features, extra], axis=1)
    candidate = metrics["selected_candidate"]
    if seed not in [entry["seed"] for entry in metrics["test"]]:
        raise ValueError("Seed not recorded in experiment")
    model_path = experiment / f"candidate_{candidate}_seed_{seed}.joblib"
    digest = sha256_file(model_path)
    model = joblib.load(model_path)
    predicted = decode(model.predict(features), tuple(layout["target_shape"]))[0]
    reference = targets[idx]
    if predicted.shape != reference.shape or not np.isfinite(predicted).all():
        raise ValueError("Model output does not match finite reference layout")
    output.mkdir(parents=True)
    prediction_figure(output / "sigma_comparison.png", mesh.imag, predicted,
                      reference, 1., "unverified", layout["target_blocks"],
                      context=f"Exploratory single run: {roles[0]} group; convergence unverified")
    summary = {
        "source": source, "source_sha256": record["source_sha256"],
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "prepared_arrays_sha256": sha256_file(arrays_path),
        "metrics_sha256": sha256_file(experiment / "metrics.json"),
        "model_sha256_observed": digest, "seed": seed, "candidate": candidate,
        "model": metrics["candidates"][candidate]["specification"],
        "split": roles[0], "group_id": record["group_id"],
        "input_path": record["input_path"], "target_path": record["target_path"],
        "rmse_complex_native": float(np.sqrt(np.mean(np.abs(predicted-reference)**2))),
        "interpretation": "Single-run exploratory inference. Train/validation/test role describes the original group split; no new independent test. Input/target meanings, units and convergence remain unresolved.",
        "provenance_note": "Model hash is observed now, not an authenticated original training hash. Only load trusted joblib files.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Render a single prepared material run with the selected existing model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Exact relative source path in manifest")
    parser.add_argument("--raw-file", type=Path, required=True, help="Read-only checksum verification")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = visualize(args.prepared, args.experiment, args.source, args.raw_file,
                       args.output, args.seed)
    print(f"Saved comparison; original split: {result['split']}; native RMSE: {result['rmse_complex_native']:.6g}")


if __name__ == "__main__":
    main()

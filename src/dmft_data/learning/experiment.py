"""Reproducible baseline selection with a separate, held-out final test set."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import warnings
from pathlib import Path

import joblib
import numpy as np

from ..provenance import sha256_file
from .data import load_json, write_json
from .models import Baseline, decode, encode
from .splits import aggregate_groups, blocked_partition


def error_metrics(
    predicted: np.ndarray, reference: np.ndarray, low_count=10, floor=1e-12
) -> dict:
    difference = np.abs(predicted - reference)
    per_sample = np.sqrt(np.mean(difference**2, axis=tuple(range(1, difference.ndim))))
    denominator = np.sqrt(np.mean(np.abs(reference) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
        "relative_rms": float(
            np.sqrt(np.mean(difference**2)) / max(denominator, floor)
        ),
        "imaginary_rmse": float(
            np.sqrt(np.mean((predicted.imag - reference.imag) ** 2))
        ),
        "low_frequency_rmse": float(np.sqrt(np.mean(difference[..., :low_count] ** 2))),
        "upper_window_rmse": float(np.sqrt(np.mean(difference[..., -low_count:] ** 2))),
        "per_group_rmse": per_sample.tolist(),
        "relative_rms_floor": floor,
        "low_upper_window_count": min(low_count, predicted.shape[-1]),
    }


def causality_metrics(values: np.ndarray, tolerance=1e-10) -> dict:
    matrices = np.moveaxis(values, -1, -3)
    antihermitian = (matrices - matrices.swapaxes(-1, -2).conj()) / (2j)
    eigenvalues = np.linalg.eigvalsh(antihermitian)
    return {
        "positive_imaginary_eigenvalue_fraction": float(
            np.mean(eigenvalues > tolerance)
        ),
        "maximum_positive_imaginary_eigenvalue": float(max(0, eigenvalues.max())),
        "tolerance": tolerance,
        "scope": "necessary_Matsubara_matrix_sign_check_only",
    }


def run_experiment(config: dict, root: Path) -> dict:
    prepared = (root / config["prepared"]).resolve()
    output = (root / config["experiment"]).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    manifest = load_json(prepared / "manifest.json")
    if sha256_file(prepared / "arrays.npz") != manifest.get("arrays_sha256"):
        raise ValueError("Prepared arrays checksum differs from the manifest")
    with np.load(prepared / "arrays.npz", allow_pickle=False) as stored:
        x, y, meshes = stored["inputs"], stored["targets"], stored["meshes"]
    records = manifest["records"]
    if manifest["task"] != config.get("task", "exploratory"):
        raise ValueError("Training task must match the prepared manifest task")
    if len({r.get("input_semantics", "unresolved") for r in records}) > 1:
        raise ValueError("Use separate experiments for different input semantics")
    if len(x) != len(records):
        raise ValueError("Prepared arrays and manifest sample counts differ")
    splits = blocked_partition(records, config["split"])
    # Exact paired duplicates cannot provide an independent evaluation.
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        keys_left = {records[i]["pair_payload_sha256"] for i in splits[left]}
        keys_right = {records[i]["pair_payload_sha256"] for i in splits[right]}
        if keys_left & keys_right:
            raise ValueError("Exact paired payload replica crosses split boundaries")
    # Grid information accompanies each curve, allowing explicitly conditioned beta studies.
    features = np.concatenate([encode(x), meshes.imag], axis=1)
    parameter_features = config.get("parameter_features", [])
    if set(parameter_features) & {"nwarmup", "seed", "iteration", "n_cycles", "run_id"}:
        raise ValueError("Solver settings and replica labels are not physical features")
    if parameter_features:
        parameters = np.array(
            [[r[key] for key in parameter_features] for r in records], dtype=float
        )
        if not np.isfinite(parameters).all():
            raise ValueError(
                "Parameter features must be explicitly supplied finite numbers"
            )
        features = np.concatenate([features, parameters], axis=1)
    grouped = {
        role: aggregate_groups(features, y, records, indices)
        for role, indices in splits.items()
    }
    for indices in splits.values():
        for group in {records[i]["group_id"] for i in indices}:
            member_meshes = meshes[
                [i for i in indices if records[i]["group_id"] == group]
            ]
            if not all(np.array_equal(member_meshes[0], m) for m in member_meshes):
                raise ValueError("Cannot average replicas on different meshes")
            members = [records[i] for i in indices if records[i]["group_id"] == group]
            for key in ["material", "beta", "input_semantics", *parameter_features]:
                if len({str(r.get(key)) for r in members}) != 1:
                    raise ValueError(f"Inconsistent {key} within replica group {group}")
    train_x, train_y, _ = grouped["train"]
    validation_x, validation_y, _ = grouped["validation"]
    candidates = config.get(
        "models",
        [
            {"name": "mean"},
            {"name": "nearest"},
            {"name": "pca_ridge", "latent_dimension": 4, "alpha": 1.0},
        ],
    )
    if not candidates:
        raise ValueError("Specify at least one model")
    seeds = config.get("seeds", [0])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify distinct seeds")
    evaluation = config.get("evaluation", {})
    metric_options = {
        "low_count": evaluation.get("low_frequency_count", 10),
        "floor": evaluation.get("relative_rms_floor", 1e-12),
    }
    if (
        type(metric_options["low_count"]) is not int
        or metric_options["low_count"] < 1
        or not np.isfinite(metric_options["floor"])
        or metric_options["floor"] <= 0
    ):
        raise ValueError(
            "Evaluation frequency count and finite RMS floor must be positive"
        )
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(
        output / "feature_layout.json",
        {
            "input_shape": list(x.shape[1:]),
            "target_shape": list(y.shape[1:]),
            "input_blocks": manifest["input_blocks"],
            "target_blocks": manifest["target_blocks"],
            "order": "flattened_complex_input_real_then_imag_then_positive_omega_then_parameters",
            "parameter_features": parameter_features,
        },
    )
    write_json(
        output / "splits.json",
        {
            role: {
                "indices": indices,
                "groups": grouped[role][2],
                "sources": [records[i]["relative_source_path"] for i in indices],
            }
            for role, indices in splits.items()
        },
    )
    scores, fitted = [], []
    for index, specification in enumerate(candidates):
        seed_scores, models = [], []
        for seed in seeds:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = Baseline(specification, seed).fit(train_x, encode(train_y))
            model.task = manifest["task"]
            model.input_semantics = records[0].get("input_semantics", "unresolved")
            model.target_shape = y.shape[1:]
            model.training_groups = grouped["train"][2]
            prediction = decode(model.predict(validation_x), y.shape[1:])
            metrics = error_metrics(prediction, validation_y, **metric_options)
            if model.pca is not None:
                projected = model.pca.inverse_transform(
                    model.pca.transform(model.y_scaler.transform(encode(validation_y)))
                )
                reconstructed = decode(
                    model.y_scaler.inverse_transform(projected), y.shape[1:]
                )
                metrics["representation_only_rmse"] = error_metrics(
                    reconstructed, validation_y, **metric_options
                )["rmse"]
            joblib.dump(model, output / f"candidate_{index}_seed_{seed}.joblib")
            seed_scores.append(
                {
                    "seed": seed,
                    "validation": metrics,
                    "actual_latent_dimension": getattr(model, "actual_dimension", None),
                    "warnings": [str(item.message) for item in caught],
                }
            )
            models.append(model)
        scores.append(
            {
                "specification": specification,
                "seeds": seed_scores,
                "validation_rmse": float(
                    np.mean([s["validation"]["rmse"] for s in seed_scores])
                ),
            }
        )
        fitted.append(models)
    selected = min(range(len(scores)), key=lambda i: scores[i]["validation_rmse"])
    # Test data are scored once for the validation-selected specification; no refit.
    test_x, test_y, test_groups = grouped["test"]
    test_mesh = np.stack(
        [
            meshes[next(i for i in splits["test"] if records[i]["group_id"] == group)]
            for group in test_groups
        ]
    )
    tests = []
    for seed, model in zip(seeds, fitted[selected]):
        prediction = decode(model.predict(test_x), y.shape[1:])
        np.savez_compressed(
            output / f"test_predictions_seed_{seed}.npz",
            predicted=prediction,
            reference=test_y,
            groups=np.asarray(test_groups),
            meshes=test_mesh,
        )
        tests.append(
            {
                "seed": seed,
                "errors": error_metrics(prediction, test_y, **metric_options),
                "prediction_causality": causality_metrics(prediction),
                "reference_causality": causality_metrics(test_y),
            }
        )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            )
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        dirty = None
    report = {
        "status": "completed",
        "task": manifest["task"],
        "interpretation": manifest["interpretation"],
        "aggregation": "arithmetic_mean_of_runs_per_reviewed_group",
        "declared_assumptions": config.get("assumptions", []),
        "input_semantics": sorted(
            {r.get("input_semantics", "unresolved") for r in records}
        ),
        "sample_count": len(records),
        "group_count": manifest["group_count"],
        "candidates": scores,
        "selected_candidate": selected,
        "test_groups": test_groups,
        "test": tests,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "prepared_arrays_sha256": sha256_file(prepared / "arrays.npz"),
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "h5py", "scikit-learn", "joblib"]
        },
        "limitations": [
            "Group counts depend on supplied metadata; uncertainty is not a QMC error bar.",
            "Sign checks alone do not establish causality, spectral accuracy, or filling.",
            "Real-frequency spectra and lattice DOS are not evaluated.",
        ],
    }
    write_json(output / "metrics.json", report)
    return report

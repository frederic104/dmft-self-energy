"""Evidence-aware diagnostics of stored Matsubara predictions and selected runs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

from .hdf import read_quantity
from .iteration_analysis import change_figure, metadata_candidates, write_changes_csv
from .learning.data import load_json, write_json
from .provenance import sha256_file

DEFINITION_FIELDS = (
    "energy_unit",
    "beta_unit",
    "input_semantics",
    "target_semantics",
    "basis",
    "chemical_potential",
    "double_counting",
    "interaction_U_J",
    "filling",
    "phase_and_symmetry",
)


def reviewed(entry: dict) -> bool:
    """A value requires a non-empty, explicitly reviewed evidence reference."""
    return (
        isinstance(entry, dict)
        and entry.get("status") == "reviewed"
        and entry.get("value") is not None
        and isinstance(entry.get("evidence"), str)
        and bool(entry["evidence"].strip())
    )


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(value) ** 2)))


def decompose(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """Frequency-constant REAL error per matrix element; never a Hartree estimate."""
    delta = predicted - reference
    offset = delta.real.mean(axis=-1, keepdims=True)
    centered = delta - offset
    return {
        "complex_rmse": rms(delta),
        "real_rmse": rms(delta.real),
        "imaginary_rmse": rms(delta.imag),
        "real_offset_rms": rms(offset),
        "signed_real_offset": offset[..., 0].tolist(),
        "real_shape_rmse": rms(centered.real),
        "oracle_centered_complex_rmse": rms(centered),
        "offset_share_total_squared_error": rms(offset) ** 2
        / max(rms(delta) ** 2, 1e-30),
    }


def _nearest_groups(
    prepared: Path, experiment: Path, metrics: dict, manifest: dict
) -> list:
    """Recover neighbor identities in the checkpoint's standardized feature space."""
    selected = metrics["selected_candidate"]
    if metrics["candidates"][selected]["specification"]["name"] != "nearest":
        return []
    import joblib

    from .learning.models import encode
    from .learning.splits import aggregate_groups

    splits = load_json(experiment / "splits.json")
    layout = load_json(experiment / "feature_layout.json")
    with np.load(prepared / "arrays.npz", allow_pickle=False) as arrays:
        features = np.concatenate(
            [encode(arrays["inputs"]), arrays["meshes"].imag], axis=1
        )
        y = arrays["targets"]
    records = manifest["records"]
    keys = layout["parameter_features"]
    if keys:
        features = np.concatenate(
            [features, np.asarray([[r[k] for k in keys] for r in records])], axis=1
        )
    _, _, train_groups = aggregate_groups(
        features, y, records, splits["train"]["indices"]
    )
    x, _, test_groups = aggregate_groups(
        features, y, records, splits["test"]["indices"]
    )
    seed = metrics["test"][0]["seed"]
    model = joblib.load(experiment / f"candidate_{selected}_seed_{seed}.joblib")
    distances, neighbors = model.model.kneighbors(
        model.x_scaler.transform(x), n_neighbors=min(3, len(train_groups))
    )
    return [
        {
            "test_group": group,
            "neighbors": [
                {"group": train_groups[j], "standardized_feature_distance": float(d)}
                for d, j in zip(distances[i], neighbors[i])
            ],
        }
        for i, group in enumerate(test_groups)
    ]


def _plot_prediction(path, omega, predicted, reference, scale, unit, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for block in range(len(reference)):
        # Diagonal curves only for display; numerical metrics include all elements.
        for orbital in range(reference.shape[1]):
            ref = reference[block, orbital, orbital]
            pred = predicted[block, orbital, orbital]
            for row, component in enumerate((np.real, np.imag)):
                color = f"C{block % 10}"
                axes[row, 0].plot(
                    omega,
                    component(ref) * scale,
                    alpha=0.7,
                    color=color,
                    label=f"{labels[block]}:{orbital}",
                )
                axes[row, 0].plot(
                    omega, component(pred) * scale, "--", alpha=0.7, color=color
                )
                axes[row, 1].plot(
                    omega, component(pred - ref) * scale, alpha=0.7, color=color
                )
    axes[0, 0].legend(fontsize=6, ncol=2)
    axis_unit = "meV" if unit == "meV" else "native"
    for row, name in enumerate(("Re", "Im")):
        axes[row, 0].set_ylabel(f"{name} value [{axis_unit}]")
        axes[row, 1].set_ylabel(f"{name} error [{axis_unit}]")
        axes[row, 1].axhline(0, color="black", linewidth=0.5)
    for ax in axes[-1]:
        ax.set_xlabel(f"positive Matsubara omega [{axis_unit}]")
    fig.suptitle(
        "Solid: reference; dashed: prediction. Diagonal elements.\n"
        f"Unit convention: {unit}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _same_layout(left, right):
    return (
        left.blocks == right.blocks
        and left.values.shape == right.values.shape
        and np.array_equal(left.mesh, right.mesh)
    )


def inspect_raw(path: Path, count: int, output: Path, stem: str) -> dict:
    """Iteration trajectories and within-run spin differences; no convergence label."""
    rows, spin = [], []
    block_changes = []
    curves = []
    with h5py.File(path, "r") as handle:
        dmft = handle.get("dmft_output")
        if not isinstance(dmft, h5py.Group):
            raise TypeError("Missing /dmft_output")
        metadata = metadata_candidates(handle)
        for quantity in ("G", "Sigma"):
            final = read_quantity(handle, f"/dmft_output/{quantity}_iw", count)
            previous = None
            previous_number = None
            names = sorted(
                (int(m.group(1)), name)
                for name in dmft
                if (m := re.fullmatch(rf"{quantity}-(\d+)", name))
            )
            for number, name in names:
                iteration = read_quantity(handle, f"/dmft_output/{name}", count)
                if not _same_layout(final, iteration):
                    raise ValueError(f"Incompatible mesh/block layout: {name}")
                rows.append(
                    {
                        "quantity": quantity,
                        "iteration": number,
                        "rms_to_final": rms(iteration.values - final.values),
                        "exact_final": bool(
                            np.array_equal(iteration.values, final.values)
                        ),
                        "rms_to_previous": rms(iteration.values - previous)
                        if previous is not None
                        else None,
                        "previous_stored_iteration": previous_number,
                        "change_from_final": decompose(iteration.values, final.values),
                        "change_from_previous": decompose(iteration.values, previous)
                        if previous is not None
                        else None,
                    }
                )
                if previous is not None:
                    for index, block in enumerate(iteration.blocks):
                        block_changes.append(
                            {
                                "quantity": quantity,
                                "block": block,
                                "from_iteration": previous_number,
                                "to_iteration": number,
                                **decompose(iteration.values[index], previous[index]),
                            }
                        )
                previous = iteration.values
                previous_number = number
                if quantity == "Sigma":
                    curves.append((number, iteration))
            for index, block in enumerate(final.blocks):
                other = "up_" + block[5:] if block.startswith("down_") else None
                if other in final.blocks:
                    spin.append(
                        {
                            "quantity": quantity,
                            "blocks": [block, other],
                            "rms_difference": rms(
                                final.values[index]
                                - final.values[final.blocks.index(other)]
                            ),
                            "interpretation": "descriptive_only_symmetry_not_assumed",
                        }
                    )
    if curves:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for number, curve in curves:
            # Each iteration displays mean diagonal Sigma; no spin symmetry inferred.
            diagonal = np.diagonal(curve.values, axis1=1, axis2=2).mean(axis=(0, 2))
            axes[0].plot(curve.mesh.imag, diagonal.real, label=str(number))
            axes[1].plot(curve.mesh.imag, diagonal.imag)
        axes[0].legend(title="iteration", fontsize=6, ncol=3)
        for ax, component in zip(axes, ("Re", "Im")):
            ax.set_xlabel("omega [native unit]")
            ax.set_ylabel(f"mean diagonal {component} Sigma [native unit]")
        fig.tight_layout()
        fig.savefig(output / f"{stem}_iterations.png", dpi=150)
        plt.close(fig)
    change_figure(output / f"{stem}_step_changes.png", rows, block_changes)
    write_changes_csv(output / f"{stem}_step_changes.csv", block_changes)
    return {
        "source_sha256": sha256_file(path),
        "iteration_comparisons": rows,
        "spin_comparisons": spin,
        "block_iteration_changes": block_changes,
        "metadata_candidates": metadata,
        "metadata_interpretation": "path/name evidence only; values do not establish restart, mixing, or physical convergence",
        "convergence_status": "not_assessed",
    }


def dyson_check(path: Path, specification: dict, count: int) -> dict:
    """Check inverse Weiss - inverse impurity G - solver Sigma at reviewed stage."""
    requirements = {
        "weiss_semantics": "dmft_weiss",
        "green_semantics": "impurity_green",
        "sigma_semantics": "impurity_self_energy",
        "sigma_convention": "solver_without_double_counting_subtraction",
    }
    evidence = specification.get("same_stage_evidence")
    if (
        any(specification.get(k) != v for k, v in requirements.items())
        or not isinstance(evidence, str)
        or not evidence.strip()
        or specification.get("source_sha256") != sha256_file(path)
    ):
        return {
            "status": "not_assessed",
            "reason": "Need scoped source, semantics and same-stage evidence",
        }
    with h5py.File(path, "r") as handle:
        weiss, green, sigma = [
            read_quantity(handle, specification[key], count)
            for key in ("weiss_path", "green_path", "sigma_path")
        ]
    if not _same_layout(weiss, green) or not _same_layout(green, sigma):
        raise ValueError("Dyson check needs identical meshes and block layouts")
    w, g, s = [np.moveaxis(q.values, -1, -3) for q in (weiss, green, sigma)]
    residual = np.linalg.inv(w) - np.linalg.inv(g) - s
    return {
        "status": "evaluated",
        "rms_residual": rms(residual),
        "max_abs_residual": float(np.abs(residual).max()),
        "specification": specification,
        "convergence_status": "not_assessed",
    }


def diagnose(
    config: dict,
    root: Path,
    definitions_path: Path | None = None,
    output: Path | None = None,
    raw_runs: list[str] | None = None,
    raw_count: int | None = None,
) -> dict:
    """Bundle saved prediction diagnostics and optionally explicit raw-run checks."""
    prepared, experiment = root / config["prepared"], root / config["experiment"]
    output = output or experiment / "physics_diagnostics"
    if output.exists():
        raise FileExistsError(f"Choose a new diagnostic output directory: {output}")
    manifest = load_json(prepared / "manifest.json")
    metrics = load_json(experiment / "metrics.json")
    if metrics["prepared_manifest_sha256"] != sha256_file(
        prepared / "manifest.json"
    ) or metrics["prepared_arrays_sha256"] != sha256_file(prepared / "arrays.npz"):
        raise ValueError("Experiment and prepared artifact checksums differ")
    review = (
        yaml.safe_load(definitions_path.read_text(encoding="utf-8"))
        if definitions_path
        else {}
    )
    review = review or {}
    definitions = review.get("definitions", {})
    unresolved = [
        key for key in DEFINITION_FIELDS if not reviewed(definitions.get(key, {}))
    ]
    per_run = {
        r["relative_source_path"]: {
            **definitions,
            **review.get("per_run", {}).get(r["relative_source_path"], {}),
        }
        for r in manifest["records"]
    }
    unit_values = [d.get("energy_unit", {}) for d in per_run.values()]
    known_units = {str(d["value"]) for d in unit_values if reviewed(d)}
    unit = (
        next(iter(known_units))
        if len(known_units) == 1 and all(reviewed(d) for d in unit_values)
        else None
    )
    scale = {"eV": 1000.0, "meV": 1.0}.get(unit, 1.0)
    display_unit = (
        "meV" if unit in {"eV", "meV"} else f"native ({unit or 'unverified'})"
    )
    output.mkdir(parents=True)
    blocks, seeds = [], []
    neighbors = _nearest_groups(prepared, experiment, metrics, manifest)
    for entry in metrics["test"]:
        seed = entry["seed"]
        with np.load(
            experiment / f"test_predictions_seed_{seed}.npz", allow_pickle=False
        ) as saved:
            predicted, reference, meshes = (
                saved["predicted"],
                saved["reference"],
                saved["meshes"],
            )
            groups = saved["groups"].tolist()
        seeds.append({"seed": seed, **decompose(predicted, reference)})
        for i, group in enumerate(groups):
            _plot_prediction(
                output / f"seed_{seed}_group_{i}.png",
                meshes[i].imag * scale,
                predicted[i],
                reference[i],
                scale,
                display_unit,
                manifest["target_blocks"],
            )
            for b, block in enumerate(manifest["target_blocks"]):
                blocks.append(
                    {
                        "seed": seed,
                        "group": group,
                        "block": block,
                        **decompose(predicted[i, b], reference[i, b]),
                    }
                )
    with (output / "block_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
    raw = []
    count = raw_count if raw_count is not None else manifest["positive_frequency_count"]
    if type(count) is not int or count < 1:
        raise ValueError("raw_count must be a positive integer")
    for i, relative in enumerate(raw_runs or []):
        source = (root / config["source"]).resolve()
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Raw-run path escapes source directory")
        matching = [
            r for r in manifest["records"] if r["relative_source_path"] == relative
        ]
        if not matching or matching[0]["source_sha256"] != sha256_file(path):
            raise ValueError("Raw run absent from manifest or source checksum changed")
        try:
            detail = inspect_raw(path, count, output, f"raw_{i}")
            detail["dyson"] = [
                dyson_check(path, spec, count)
                for spec in review.get("dyson_checks", [])
                if spec.get("relative_source_path") == relative
            ]
            raw.append({"relative_source_path": relative, **detail})
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            np.linalg.LinAlgError,
        ) as error:
            raw.append(
                {
                    "relative_source_path": relative,
                    "status": "failed",
                    "error": str(error),
                }
            )
    report = {
        "task": metrics["task"],
        "convergence_status": "not_assessed",
        "definitions": definitions,
        "unresolved_definitions": unresolved,
        "per_run_definitions": per_run,
        "per_run_unresolved_definitions": {
            path: [
                key for key in DEFINITION_FIELDS if not reviewed(fields.get(key, {}))
            ]
            for path, fields in per_run.items()
        },
        "definitions_sha256": sha256_file(definitions_path)
        if definitions_path
        else None,
        "metrics_sha256": sha256_file(experiment / "metrics.json"),
        "native_energy_unit": unit,
        "plot_unit": display_unit,
        "metric_unit": "native",
        "nearest_groups": neighbors,
        "seeds": seeds,
        "blocks": blocks,
        "raw_runs": raw,
        "deferred": [
            "high_frequency_moments",
            "filling",
            "Z",
            "spectral_function",
            "lattice_DOS",
        ],
        "interpretation": "Oracle-centered errors use the reference and are not predictive performance. Window means are not Hartree moments. No universal energy tolerance applied.",
    }
    write_json(output / "diagnostics.json", report)
    lines = [
        "# Physics diagnostic report",
        "",
        f"Task: {metrics['task']}",
        "Convergence: not_assessed",
        f"Plot units: {display_unit}; metric units: native.",
        "",
        "## Definition evidence",
        "",
        "| Field | Value | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for key in DEFINITION_FIELDS:
        item = definitions.get(key, {})
        lines.append(
            f"| {key} | {item.get('value')} | {'reviewed' if reviewed(item) else 'unresolved'} | {item.get('evidence')} |"
        )
    lines += ["", "## Stored prediction errors", ""]
    for neighbor in neighbors:
        lines.append(
            f"- {neighbor['test_group']} nearest: {neighbor['neighbors'][0]['group']} (standardized input space)."
        )
    for entry in seeds:
        lines.append(
            f"- Seed {entry['seed']}: RMSE={entry['complex_rmse']:.6g}; oracle-centered={entry['oracle_centered_complex_rmse']:.6g} (native units)."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "Raw iteration differences and spin differences are descriptive; no symmetry or convergence certificate is inferred.",
        "Total-energy SCF tolerances are not self-energy RMSE thresholds.",
        "See diagnostics.json for requested raw-run failures and deferred physical tests.",
    ]
    lines += ["", "## Raw-run change review", ""]
    for run in raw:
        lines.append(f"### {run['relative_source_path']}")
        if run.get("status") == "failed":
            lines.append(f"\nFailed: {run['error']}\n")
            continue
        steps = sorted(
            (
                r
                for r in run["iteration_comparisons"]
                if r["quantity"] == "Sigma" and r["change_from_previous"] is not None
            ),
            key=lambda r: r["change_from_previous"]["complex_rmse"],
            reverse=True,
        )
        lines += [
            "",
            "Largest stored Sigma changes (native units; descriptive ranking):",
            "",
        ]
        for step in steps[:3]:
            delta = step["change_from_previous"]
            lines.append(
                f"- {step['previous_stored_iteration']} -> {step['iteration']}: "
                f"total={delta['complex_rmse']:.6g}, Re constant={delta['real_offset_rms']:.6g}, "
                f"Re shape={delta['real_shape_rmse']:.6g}, Im={delta['imaginary_rmse']:.6g}."
            )
        lines += [
            "",
            (
                f"Metadata candidates found: {len(run['metadata_candidates'])}. "
                "See JSON for exact paths and values. Absence does not prove a parameter was absent from the calculation."
            ),
            "",
        ]
    lines += ["", "## Figures", ""]
    for image in sorted(output.glob("*.png")):
        lines.append(f"![{image.stem}]({image.name})")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

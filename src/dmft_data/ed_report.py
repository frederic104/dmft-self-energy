"""Report existing AIM results and saved predictions; never load models or train."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from matplotlib.figure import Figure

from .ed_properties import compare_properties, dyson_errors, occupancy_estimates
from .ed_spectra import integrate, spectrum_metrics
from .provenance import sha256_file


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rmse(x, axis=None):
    return np.sqrt(np.mean(np.abs(x)**2, axis=axis))


def state_key(parameters):
    # Identifies the complete physical tuple, independent of row order or seed.
    return hashlib.sha256(np.asarray(parameters, dtype="<f8").tobytes()).hexdigest()


def aggregate(rows):
    """Mean/SD of seed metrics, never metrics of an ensemble prediction."""
    groups = defaultdict(list)
    identifiers = ("run", "phase", "axis", "candidate", "eta")
    for row in rows:
        groups[tuple(row.get(k, "") for k in identifiers)].append(row)
    result = []
    for key, group in groups.items():
        good = [r for r in group if r["status"] == "ok"]
        metrics = sorted({k for r in good for k, v in r.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)
                          and k not in {"model_seed", "eta", "optimizer_iterations"}})
        for metric in metrics:
            values = [r[metric] for r in good if r.get(metric) is not None]
            result.append(dict(zip(identifiers, key), metric=metric,
                               count=len(values), expected_rows=len(group),
                               mean=float(np.mean(values)),
                               std=float(np.std(values, ddof=0)) if len(values) > 1 else None))
    return result


def check_metrics(saved, actual):
    for key, value in actual.items():
        if (key in saved and isinstance(value, (float, int))
                and not np.isclose(value, saved[key], rtol=1e-7, atol=1e-11)):
            raise ValueError(f"Saved score differs from arrays: {key}")


def frozen_diagnostics(root, summary, rows, record, missing):
    states, frequencies, diagnostics = [], [], []
    for axis, expected_hash in summary["dataset_sha256"].items():
        dataset = root / axis / "dataset.npz"
        if not dataset.exists():
            missing.append(str(dataset))
            continue
        if record(dataset) != expected_hash:
            raise ValueError(f"Dataset checksum mismatch: {dataset}")
        with np.load(dataset, allow_pickle=False) as data:
            d = {k: data[k] for k in data.files}
        params, omega = d["parameters"], d["omega"]
        if not np.all(params[:, 7] == params[0, 7]):
            raise ValueError("Frozen diagnostics require fixed beta")
        beta = float(params[0, 7])
        for seed in dict.fromkeys(r["model_seed"] for r in rows if r["axis"] == axis):
            path = root / axis / f"predictions_seed_{seed}.npz"
            if not path.exists():
                missing.append(str(path))
                continue
            record(path)
            with np.load(path, allow_pickle=False) as data:
                a = {k: data[k] for k in data.files}
            for k, target in (("parameters", "parameters"), ("omega", "omega"),
                              ("reference_sigma", "sigma"), ("reference_green", "green"),
                              ("reference_occupancy_per_spin", "occupancy_per_spin")):
                np.testing.assert_array_equal(a[k], d[target])
            for row in [r for r in rows if r["axis"] == axis and r["model_seed"] == seed
                        and r["status"] == "ok"]:
                name = row["candidate"]
                keys = [f"{name}_{k}" for k in ("sigma", "green", "occupancy_tail")]
                if any(k not in a for k in keys):
                    missing.append(f"{path}: {name} arrays")
                    continue
                sigma, green, n = (a[k] for k in keys)
                scores, recomputed = compare_properties(
                    d["g0"], sigma, d["green"], d["occupancy_per_spin"], omega, beta)
                np.testing.assert_allclose(green, recomputed["green"], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(n, recomputed["occupancy_tail"], rtol=1e-10, atol=1e-12)
                scores["sigma_rmse"] = float(rmse(sigma-d["sigma"]))
                check_metrics(row, scores)
                e = dyson_errors(d["sigma"], sigma, d["green"], green)
                ids = {k: row[k] for k in ("run", "phase", "axis", "model_seed", "candidate")}
                sg, gg = np.abs(e["sigma_error"])**2, np.abs(e["green_error"])**2
                gnorm = float(rmse(e["green_error"]))
                diagnostics.append({**ids, "state_count": len(params),
                    "identity_residual_max": float(np.max(np.abs(e["identity_residual"]))),
                    "linear_relative_l2_remainder": float(rmse(e["linear_remainder"])/gnorm) if gnorm else None,
                    "expansion_parameter_max": float(e["expansion_parameter"].max()),
                    "expansion_parameter_p95": float(np.quantile(e["expansion_parameter"], .95)),
                    "sigma_first_8_squared_error_share": float(sg[:, :8].sum()/sg.sum()) if sg.sum() else None,
                    "green_first_8_squared_error_share": float(gg[:, :8].sum()/gg.sum()) if gg.sum() else None,
                    **{k: scores[k] for k in ("sigma_rmse", "green_first_8_complex_rmse",
                        "occupancy_tail_vs_exact_rmse", "reference_tail_vs_exact_rmse")}})
                _, ref_n = occupancy_estimates(d["green"], omega, beta)
                for i, p in enumerate(params):
                    states.append({**ids, "state_id": state_key(p), "state_index": i,
                        "source_index": int(d.get("source_indices", np.arange(len(params)))[i]),
                        "epsilon_d": float(p[0]), "ed_occupancy_per_spin": float(d["occupancy_per_spin"][i]),
                        "sigma_rmse": float(rmse(e["sigma_error"][i])),
                        "green_first_8_rmse": float(rmse(e["green_error"][i, :8])),
                        "occupancy_error": float(n[i]-d["occupancy_per_spin"][i]),
                        "occupancy_abs_error": float(abs(n[i]-d["occupancy_per_spin"][i])),
                        "occupancy_model_minus_reference_tail": float(n[i]-ref_n[i]),
                        "reference_tail_minus_exact": float(ref_n[i]-d["occupancy_per_spin"][i]),
                        "expansion_parameter_max": float(e["expansion_parameter"][i].max())})
                for j, w in enumerate(omega):
                    frequencies.append({**ids, "frequency_index": j, "omega": float(w),
                        **{k + "_rmse": float(rmse(e[k][:, j])) for k in (
                            "sigma_error", "green_error", "linear_green_error", "linear_remainder")},
                        "exact_amplification_mean": float(e["exact_amplification"][:, j].mean()),
                        "expansion_parameter_max": float(e["expansion_parameter"][:, j].max())})
    return states, frequencies, diagnostics


def spectral_diagnostics(root, summary, rows, record, missing, matsubara_root=None):
    states = []
    for axis in summary["dataset_sha256"]:
        for eta in summary["etas"]:
            path = root / axis / f"spectra_eta_{eta:g}.npz"
            if not path.exists():
                missing.append(str(path))
                continue
            record(path)
            with np.load(path, allow_pickle=False) as data:
                a = {k: data[k] for k in data.files}
            energy, ref, params = (a[k] for k in ("energy", "reference_spectrum", "parameters"))
            if a["eta"] != eta or len(energy) != summary["points"]:
                raise ValueError("Spectral grid differs from report")
            np.testing.assert_allclose(energy, np.linspace(
                -summary["energy_bound"], summary["energy_bound"], summary["points"]))
            if matsubara_root is not None:
                dataset = matsubara_root / axis / "dataset.npz"
                if dataset.exists():
                    with np.load(dataset, allow_pickle=False) as d:
                        np.testing.assert_array_equal(params, d["parameters"])
                        np.testing.assert_array_equal(a["exact_occupancy_per_spin"], d["occupancy_per_spin"])
            for row in [r for r in rows if r["axis"] == axis and r["eta"] == eta and r["status"] == "ok"]:
                prefix = f"seed_{row['model_seed']}_{row['candidate']}"
                if any(prefix + suffix not in a for suffix in (
                        "_spectrum", "_sigma_infinity", "_sigma_first_moment")):
                    missing.append(f"{path}: {prefix} arrays")
                    continue
                predicted = a[prefix + "_spectrum"]
                metrics, _ = spectrum_metrics(predicted, ref, energy, params[:, 7], a["exact_occupancy_per_spin"])
                for suffix, target in (("sigma_infinity", params[:, 1]*a["exact_occupancy_per_spin"]),
                                       ("sigma_first_moment", params[:, 1]**2*a["exact_occupancy_per_spin"]*(1-a["exact_occupancy_per_spin"]))):
                    metrics[suffix + "_rmse"] = float(rmse(a[prefix + "_" + suffix]-target))
                check_metrics(row, metrics)
                for i, p in enumerate(params):
                    states.append({**{k: row[k] for k in ("run", "phase", "axis", "model_seed", "candidate", "eta")},
                        "state_id": state_key(p), "state_index": i, "epsilon_d": float(p[0]),
                        "ed_occupancy_per_spin": float(a["exact_occupancy_per_spin"][i]),
                        "spectrum_rmse": float(rmse(predicted[i]-ref[i])),
                        "integrated_absolute_error": float(integrate(np.abs(predicted[i]-ref[i]), energy))})
    return states


def plot_states(states, frequencies, output):
    groups = defaultdict(list)
    for row in states:
        groups[(row["run"], row["axis"], row.get("eta", ""))].append(row)
    for number, ((run, axis, eta), group) in enumerate(groups.items()):
        spectral = eta != ""
        fig = Figure(figsize=(12, 7), layout="constrained")
        axes = fig.subplots(2, 3)
        keys = (["spectrum_rmse", "integrated_absolute_error"] if spectral else
                ["sigma_rmse", "green_first_8_rmse", "occupancy_abs_error"])
        for name in dict.fromkeys(r["candidate"] for r in group):
            physical = defaultdict(list)
            for r in group:
                if r["candidate"] == name:
                    physical[r["state_id"]].append(r)
            # One plotted point per physical state; seeds remain separate in CSV.
            means = [{k: np.mean([r[k] for r in rs]) for k in
                      ["epsilon_d", "ed_occupancy_per_spin", *keys]} for rs in physical.values()]
            label = name.replace("_latent_8_refine_", " ").replace("_ridge_8", "8")
            for j, key in enumerate(keys):
                x = "epsilon_d" if j == 0 else "ed_occupancy_per_spin"
                axes[0, j].scatter([r[x] for r in means], [r[key] for r in means], s=12, label=label)
                axes[0, j].set(xlabel=x, ylabel=key)
                if spectral:
                    values = sorted(r[key] for r in means)
                    axes[1, j].step(values, np.arange(1, len(values)+1)/len(values), label=label)
                    axes[1, j].set(xlabel=key, ylabel="All-state ECDF")
            if not spectral:
                freq = defaultdict(list)
                for r in frequencies:
                    if r["run"] == run and r["axis"] == axis and r["candidate"] == name:
                        freq[r["omega"]].append(r)
                for j, key in enumerate(("sigma_error_rmse", "green_error_rmse", "expansion_parameter_max")):
                    xs = sorted(freq)
                    axes[1, j].plot(xs, [np.mean([r[key] for r in freq[w]]) for w in xs], label=label)
                    axes[1, j].set(xscale="log", xlabel="Matsubara frequency", ylabel=key)
        if spectral:
            axes[0, 2].axis("off")
            axes[1, 2].axis("off")
        axes[0, 0].legend(fontsize="x-small")
        fig.suptitle(f"{run}: {axis}" + (f", eta={eta}" if spectral else "") +
                     "\nAll states; mean of seed diagnostics, not ensemble predictions")
        fig.savefig(output / f"states_{number:02d}.png", dpi=150)


def report(results, output):
    """Consolidate existing summary/NPZ files, preserving phase and failed rows."""
    results, output = [Path(p) for p in results], Path(output)
    if output.exists():
        raise FileExistsError(output)
    if len({p.name for p in results}) != len(results):
        raise ValueError("Result directory names must be distinct")
    artifacts, missing, rows, states, frequencies, diagnostics, spectral, termination = {}, [], [], [], [], [], [], []

    def label(path):
        for root in results:
            if path.is_relative_to(root):
                return root.name + "/" + path.relative_to(root).as_posix()
        raise ValueError("Artifact outside supplied result directories")

    def record(path):
        digest = sha256_file(path)
        artifacts[label(path)] = digest
        return digest

    summaries = {}
    for root in results:
        path = root / "summary.json"
        record(path)
        summaries[root] = json.loads(path.read_text(encoding="utf-8"))
    for root, s in summaries.items():
        phase = s.get("phase", s.get("recipe", {}).get("evaluation_phase", "unknown"))
        raw = list(s.get("comparisons", [])) + list(s.get("failures", []))
        if "complex_rmse" in s:
            raw = [{"candidate": n, "sigma_rmse": v} for n, v in s["complex_rmse"].items()]
        local = [{**r, "run": root.name, "phase": phase,
                  "candidate": r.get("candidate", "representation_" + r.get("pole_spacing", "unknown")),
                  "status": r.get("status", r.get("property_status", "failed" if
                             r.get("representation_fit_success") is False else "ok"))} for r in raw]
        rows.extend(local)
        for case in s.get("protocol", {}).get("cases", []):
            for name, info in case.get("training_termination", {}).items():
                termination.append({"run": root.name, "axis": case["axis"], "model_seed": case["model_seed"],
                                    "candidate": name, **{k: v for k, v in info.items() if k != "history"}})
        for path in sorted(root.glob("*/seed_*/metrics.json")):
            record(path)
            m = json.loads(path.read_text())
            design = m.get("decoder_design", {})
            if design:
                rows.append({"run": root.name, "phase": "training_representation_only",
                             "axis": path.parent.parent.name,
                             "model_seed": int(path.parent.name.split("_")[-1]),
                             "candidate": "representation_" + design.get("pole_spacing", "uniform"),
                             "status": "ok" if design["representation_fit_success"] else "failed",
                             "training_representation_rmse": design["training_representation_rmse"]})
            for name, info in m.get("refinement", {}).items():
                termination.append({"run": root.name, "axis": path.parent.parent.name,
                                    "model_seed": int(path.parent.name.split("_")[-1]), "candidate": name,
                                    **{k: v for k, v in info.items() if k != "history"}})
        if phase in {"frozen_reserved_scale", "frozen_unseen_epsilon"}:
            st, fr, dg = frozen_diagnostics(root, s, local, record, missing)
            states.extend(st)
            frequencies.extend(fr)
            diagnostics.extend(dg)
        if phase == "frozen_impurity_spectra":
            linked = [(p, t) for p, t in summaries.items() if artifacts[label(p/"summary.json")] == s.get("matsubara_report_sha256")]
            if not linked:
                missing.append(f"{root}: linked Matsubara summary not supplied")
            elif (linked[0][1]["dataset_sha256"] != s["dataset_sha256"]
                  or linked[0][1]["protocol_sha256"] != s["protocol_sha256"]
                  or linked[0][1]["model_bundle_sha256"] != s["bundle_sha256"]):
                raise ValueError("Spectral/Matsubara provenance mismatch")
            spectral.extend(spectral_diagnostics(root, s, local, record, missing,
                                                 linked[0][0] if linked else None))
    # Keep matched state/seed units, without pooling 3 seeds into 120 states.
    baseline = {(r["run"], r["axis"], r["model_seed"], r["state_id"]): r for r in states
                if r["candidate"] == "pca_ridge_8"}
    paired = []
    for r in states:
        b = baseline.get((r["run"], r["axis"], r["model_seed"], r["state_id"]))
        if b is not None and r["candidate"] != "pca_ridge_8":
            paired.append({**r, **{f"delta_{k}_squared_vs_pca8": r[k]**2-b[k]**2
                                  for k in ("sigma_rmse", "green_first_8_rmse", "occupancy_error")}})
    for root in results:
        missing = [message.replace(str(root), root.name) for message in missing]
    output.mkdir(parents=True)
    grouped = aggregate(rows)
    for name, data in (("comparisons", rows), ("metric_summary", grouped), ("state_errors", states),
                       ("frequency_errors", frequencies), ("dyson_diagnostics", diagnostics),
                       ("paired_state_errors", paired), ("spectral_state_errors", spectral),
                       ("training_termination", termination)):
        write_csv(output / f"{name}.csv", data)
    plot_states(states + spectral, frequencies, output)
    summary = {"phase": "posthoc_closed_study_report", "search_status": "closed",
               "artifact_sha256": artifacts, "missing": missing,
               "status_counts": dict(Counter(r["status"] for r in rows)),
               "metric_summary": grouped, "dyson_diagnostics": diagnostics,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
               "source_sha256": {n: sha256_file(Path(__file__).parent/n) for n in ("ed_report.py", "ed_properties.py", "ed_spectra.py")},
               "note": "Post-hoc, all states; ddof=0 across seeds, null SD for one seed. No training, inference, selection or independent rerun of ED. New hashes record files read, not original prediction authentication."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = ["# Saved-result report", "", summary["note"], "",
            "| Run / phase | Axis | Model | eta | Metric | Mean | Population SD | Count |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    primary = {"sigma_rmse", "green_first_8_complex_rmse", "occupancy_tail_vs_exact_rmse",
               "spectrum_rmse", "mean_integrated_absolute_error", "training_representation_rmse"}
    for r in grouped:
        if r["metric"] in primary:
            sd = "unavailable" if r["std"] is None else f"{r['std']:.6g}"
            text.append(f"| {r['run']} / {r['phase']} | {r['axis']} | {r['candidate']} | {r['eta']} | {r['metric']} | {r['mean']:.6g} | {sd} | {r['count']}/{r['expected_rows']} |")
    text += ["", "All failed/unsupported rows: comparisons.csv. Budget termination: training_termination.csv.",
             "Missing inputs (never replaced by invented values): " + json.dumps(missing), ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = report(args.results, args.output)
    print(f"Saved {args.output}; missing inputs: {len(summary['missing'])}")


if __name__ == "__main__":
    main()

"""Frozen causal impurity spectra against ED; no candidate search or fitting."""

import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure
from scipy.special import expit

from .ed import (
    AIMParameters,
    impurity_occupancy,
    noninteracting_green_z,
    solve,
    spectral_function,
)
from .ed_properties import dyson_green
from .learning.causal_decoder import PoleBaseline
from .learning.models import encode
from .provenance import sha256_file


def integrate(values, energy):
    return np.sum((values[..., 1:] + values[..., :-1]) * np.diff(energy) / 2, axis=-1)


def spectrum_metrics(predicted, reference, energy, beta, exact_n):
    """Finite-window/broadened integrals are never renormalized or clipped."""
    if predicted.shape != reference.shape or not np.isfinite(predicted).all():
        raise ValueError("Invalid predicted spectra")
    weight, ref_weight = (integrate(a, energy) for a in (predicted, reference))
    occupation, ref_occupation = (integrate(a * expit(-beta[:, None] * energy), energy)
                                  for a in (predicted, reference))
    def rmse(a):
        return float(np.sqrt(np.mean(np.abs(a)**2)))
    return {"spectrum_rmse": rmse(predicted - reference),
            "mean_integrated_absolute_error": float(integrate(np.abs(predicted-reference), energy).mean()),
            "negative_spectrum_fraction": float(np.mean(predicted < -1e-10)),
            "window_weight_mean": float(weight.mean()),
            "window_weight_vs_one_rmse": rmse(weight - 1),
            "reference_window_weight_mean": float(ref_weight.mean()),
            "reference_window_weight_vs_one_rmse": rmse(ref_weight - 1),
            "occupation_vs_broadened_reference_rmse": rmse(occupation - ref_occupation),
            "occupation_vs_exact_ed_rmse": rmse(occupation - exact_n),
            "reference_broadened_occupation_vs_exact_ed_rmse": rmse(ref_occupation - exact_n)}, {
                "window_weight": weight, "occupation": occupation, "reference_window_weight": ref_weight,
                "reference_occupation": ref_occupation}


def evaluate_spectra(panel: Path, output: Path, *, energy_bound=8., points=1601, etas=(.05, .1),
                     matsubara_result: Path | None = None):
    panel, output = Path(panel), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (not np.isfinite(energy_bound) or energy_bound <= 0 or points < 3 or not etas
            or not np.isfinite(etas).all() or min(etas) <= 0 or len(set(etas)) != len(etas)):
        raise ValueError("Invalid spectral grid or broadening")
    if 2 * energy_bound / (points-1) > min(etas) / 4:
        raise ValueError("Resolve the narrowest broadening with at least four grid intervals")
    protocol = json.loads((panel / "protocol.json").read_text())
    if protocol.get("search_status") != "closed" or protocol.get("evaluation_scope") != "reserved_scale":
        raise ValueError("Use the closed final panel with its reserved states")
    if sha256_file(panel / "panel.joblib") != protocol["bundle_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    previous = None
    if matsubara_result is not None:
        previous = json.loads((Path(matsubara_result) / "summary.json").read_text())
        if (previous["phase"] != "frozen_reserved_scale"
                or previous["model_bundle_sha256"] != protocol["bundle_sha256"]
                or previous["protocol_sha256"] != sha256_file(panel / "protocol.json")
                or previous["dataset_sha256"] != protocol["reserved_datasets"]):
            raise ValueError("Matsubara report does not match this frozen panel and states")
    models = joblib.load(panel / "panel.joblib")  # Trusted local artifact only.
    energy = np.linspace(-energy_bound, energy_bound, points)
    output.mkdir(parents=True)
    rows, dataset_hashes = [], {}
    for axis in protocol["source_recipe"]["axes"]:
        path = panel / axis / "reserved.npz"
        dataset_hashes[axis] = sha256_file(path)
        if dataset_hashes[axis] != protocol["reserved_datasets"][axis]:
            raise ValueError("Reserved dataset checksum mismatch")
        with np.load(path) as data:
            params, g0_iw, exact_n = (data[k].copy() for k in ("parameters", "g0", "occupancy_per_spin"))
        parameters = [AIMParameters(p[0], p[1], tuple(p[2:4]), tuple(p[4:6]), p[6], p[7]) for p in params]
        # Solve each physical state once, reused over all seeds and broadenings.
        references = [solve(p) for p in parameters]
        np.testing.assert_allclose([impurity_occupancy(r) for r in references], exact_n, atol=1e-10, rtol=1e-10)
        target = output / axis
        target.mkdir()
        for eta in etas:
            z = energy + 1j * eta
            reference = np.stack([spectral_function(r, energy, eta) for r in references])
            g0 = np.stack([noninteracting_green_z(p, z) for p in parameters])
            arrays = {"energy": energy, "eta": eta, "parameters": params,
                      "reference_spectrum": reference, "exact_occupancy_per_spin": exact_n}
            fig = Figure(figsize=(8, 4), layout="constrained")
            ax = fig.subplots()
            ax.plot(energy, reference[0], color="black", label="ED")
            for case in [c for c in protocol["cases"] if c["axis"] == axis]:
                for name in protocol["panel"]:
                    model = models[case["key"]][name]
                    row = {"axis": axis, "model_seed": case["model_seed"], "candidate": name, "eta": eta}
                    if not isinstance(model, PoleBaseline):
                        row.update(status="unsupported", reason="Matsubara-only model; no validated real-frequency representation")
                    else:
                        try:
                            sigma = model.predict_z(encode(g0_iw), z)
                            predicted = -dyson_green(g0, sigma).imag / np.pi
                            metrics, integrals = spectrum_metrics(predicted, reference, energy, params[:, 7], exact_n)
                            constant, residues = model.pole_parameters(encode(g0_iw))
                            hartree = params[:, 1] * exact_n
                            moment1 = params[:, 1]**2 * exact_n * (1-exact_n)
                            row.update(status="ok", **metrics,
                                       sigma_infinity_rmse=float(np.sqrt(np.mean((constant-hartree)**2))),
                                       sigma_first_moment_rmse=float(np.sqrt(np.mean((residues.sum(axis=1)-moment1)**2))))
                            prefix = f"seed_{case['model_seed']}_{name}"
                            arrays[prefix + "_spectrum"] = predicted
                            ax.plot(energy, predicted[0], label=f"causal seed {case['model_seed']}", alpha=.7)
                            arrays[prefix + "_sigma_infinity"] = constant
                            arrays[prefix + "_sigma_first_moment"] = residues.sum(axis=1)
                            arrays.update({f"{prefix}_{k}": v for k, v in integrals.items()})
                        except (ValueError, FloatingPointError) as exc:
                            row.update(status="failed", reason=str(exc))
                    rows.append(row)
            np.savez_compressed(target / f"spectra_eta_{eta:g}.npz", **arrays)
            ax.set(xlabel="Energy relative to mu", ylabel="Per-spin impurity A(energy)",
                   title=f"{axis}: first reserved state (index 0), eta={eta:g}")
            ax.legend()
            fig.savefig(target / f"example_eta_{eta:g}.png", dpi=150)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"phase": "frozen_impurity_spectra", "search_status": "closed",
               "energy_bound": energy_bound, "points": points, "etas": etas,
               "protocol_sha256": sha256_file(panel / "protocol.json"),
               "bundle_sha256": protocol["bundle_sha256"], "dataset_sha256": dataset_hashes,
               "python": platform.python_version(), "numpy": np.__version__, "comparisons": rows,
               "package_versions": {name: version(name) for name in ("scipy", "scikit-learn", "matplotlib", "joblib")},
               "source_sha256": {n: sha256_file(Path(__file__).parent / n) for n in (
                   "ed_spectra.py", "ed.py", "ed_properties.py", "learning/causal_decoder.py")},
               "interpretation": "Per-spin broadened finite-bath impurity spectra, not lattice DOS. Finite-window/broadening occupation bias is reported using ED. No model selection or training; unsupported candidates are not zero-error scores."}
    if previous is not None:
        summary["matsubara_report_sha256"] = sha256_file(Path(matsubara_result) / "summary.json")
        summary["matsubara_seed_summary"] = previous["seed_summary"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--energy-bound", type=float, default=8.)
    parser.add_argument("--points", type=int, default=1601)
    parser.add_argument("--etas", type=float, nargs="+", default=[.05, .1])
    parser.add_argument("--matsubara-result", type=Path)
    args = parser.parse_args()
    evaluate_spectra(args.panel, args.output, energy_bound=args.energy_bound, points=args.points,
                     etas=tuple(args.etas), matsubara_result=args.matsubara_result)
    print(f"Saved {args.output / 'comparison.csv'}")


if __name__ == "__main__":
    main()

"""Descriptive iteration-change decomposition and bounded metadata discovery."""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np

METADATA_NAMES = {
    "mu",
    "chemicalpotential",
    "chemicalpotentialpre",
    "chemicalpotentialpost",
    "dc",
    "dcimp",
    "dcenerg",
    "doublecounting",
    "mix",
    "mixing",
    "sigmamix",
    "g0mix",
    "restart",
    "restarted",
    "iteration",
    "iterations",
    "niterations",
    "seed",
    "randomseed",
    "nwarmup",
    "nwarmupcycles",
    "ncycles",
    "filling",
    "density",
    "occupation",
    "occupations",
    "solver",
    "u",
    "jh",
    "hundj",
}


def _name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def metadata_candidates(handle: h5py.File) -> list[dict]:
    """Find exact metadata names, preserving paths; do not interpret their roles."""
    records = []

    def convert(value):
        if isinstance(value, bytes):
            return convert(value.decode("utf-8", errors="replace"))
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, (float, int, bool)):
            return value if np.isfinite(value) else str(value)
        text = str(value)
        return (
            text if len(text) <= 500 else {"text_prefix": text[:500], "truncated": True}
        )

    def visit(name, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and _name(name.rsplit("/", 1)[-1]) in METADATA_NAMES
        ):
            entry = {
                "source_path": obj.name,
                "kind": "dataset",
                "shape": list(obj.shape),
            }
            if obj.size <= 16:
                entry["value"] = convert(obj[()])
            else:
                entry["value_status"] = "not_read_large_metadata_candidate"
            records.append(entry)
        for key in obj.attrs:
            if _name(key) not in METADATA_NAMES:
                continue
            identifier = obj.attrs.get_id(key)
            if np.prod(identifier.shape, dtype=int) > 16:
                records.append(
                    {
                        "source_path": obj.name,
                        "attribute": key,
                        "value_status": "not_read_large_attribute_candidate",
                    }
                )
            else:
                records.append(
                    {
                        "source_path": obj.name,
                        "kind": "attribute",
                        "attribute": key,
                        "value": convert(obj.attrs[key]),
                    }
                )

    visit("", handle)
    handle.visititems(visit)
    return records


def change_figure(path: Path, rows: list[dict], block_changes: list[dict]) -> None:
    """Plot adjacent Sigma components and blocks, retaining zeros on linear axes."""
    from .plotting import COLORS, MARKERS, figure_style, finish

    selected = [
        r
        for r in rows
        if r["quantity"] == "Sigma" and r["change_from_previous"] is not None
    ]
    if not selected:
        return
    with figure_style() as plt:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained")
        for i, (key, label) in enumerate(
            (
                ("real_offset_rms", "constant Re"),
                ("real_shape_rmse", "varying Re"),
                ("imaginary_rmse", "Im"),
            )
        ):
            axes[0].plot(
                [r["iteration"] for r in selected],
                [r["change_from_previous"][key] for r in selected],
                color=COLORS[i],
                marker=MARKERS[i],
                markersize=3,
                linestyle=("-", "--", ":")[i],
                label=label,
            )
        blocks = sorted({r["block"] for r in block_changes if r["quantity"] == "Sigma"})
        for i, block in enumerate(blocks):
            points = [
                r
                for r in block_changes
                if r["quantity"] == "Sigma" and r["block"] == block
            ]
            axes[1].plot(
                [r["to_iteration"] for r in points],
                [r["complex_rmse"] for r in points],
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                markersize=3,
                linestyle="-" if i % 2 == 0 else "--",
                label=block,
            )
        for ax in axes:
            ax.set_xlabel("Stored iteration")
            ax.set_ylabel(r"RMS $\Delta\Sigma$ (native units)")
            ax.legend(frameon=False, ncol=2)
        finish(
            fig,
            axes,
            path,
            "(a) Frequency-constant real shift, frequency-dependent real residual and imaginary change. "
            "(b) Block-resolved complex RMS changes. Each point compares with the previous STORED iteration; "
            "missing iteration numbers are not filled. Axes are linear and include zeros. "
            "Window and signed offsets are in the companion CSV/JSON. These are descriptive changes, not convergence criteria.",
        )
        plt.close(fig)


def write_changes_csv(path: Path, changes: list[dict]) -> None:
    """Store scalar per-block changes plus signed matrix offsets in JSON cells."""
    import csv

    if not changes:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
        writer.writeheader()
        for row in changes:
            writer.writerow(
                {**row, "signed_real_offset": json.dumps(row["signed_real_offset"])}
            )

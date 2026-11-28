"""Consistent scientific figures: physical axes, panel labels, vector exports."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v", "P")


@contextmanager
def figure_style():
    """Apply local plotting parameters without changing other experiments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    ):
        yield plt


def finish(fig, axes, path: Path, caption: str) -> None:
    """Save screen preview, vector PDF, and a separate draft figure caption."""
    for index, ax in enumerate(np.asarray(axes).flat):
        ax.text(
            0.02,
            0.97,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
        )
        ax.minorticks_on()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    path.with_suffix(".caption.md").write_text(caption + "\n", encoding="utf-8")


def prediction_figure(path, omega, predicted, reference, scale, unit, labels, context=None):
    """Overview preserves each block; the companion resolves blocks separately."""
    native = "meV" if unit == "meV" else "native units"
    with figure_style() as plt:
        fig, axes = plt.subplots(
            2, 2, figsize=(7.1, 4.9), sharex=True, layout="constrained"
        )
        if context:
            fig.suptitle(context, fontsize=9)
        for b, label in enumerate(labels):
            for orbital in range(reference.shape[1]):
                ref, pred = (
                    reference[b, orbital, orbital],
                    predicted[b, orbital, orbital],
                )
                for row, component in enumerate((np.real, np.imag)):
                    args = {"color": COLORS[b % len(COLORS)]}
                    axes[row, 0].plot(
                        omega,
                        component(ref) * scale,
                        label=f"{label}:{orbital}",
                        **args,
                    )
                    axes[row, 0].plot(omega, component(pred) * scale, "--", **args)
                    axes[row, 1].plot(
                        omega,
                        component(pred - ref) * scale,
                        marker=MARKERS[b % len(MARKERS)],
                        markersize=2.5,
                        markevery=max(1, len(omega) // 8),
                        **args,
                    )
        axes[0, 0].legend(ncol=2, loc="best", frameon=False)
        for row, component in enumerate(("Re", "Im")):
            axes[row, 0].set_ylabel(f"{component} target ({native})")
            axes[row, 1].set_ylabel(f"{component} error ({native})")
            axes[row, 1].axhline(0, color="0.5", linewidth=0.6)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\omega_n$" + f" ({native})")
        caption = (
            f"(a,c) Diagonal reference (solid) and prediction (dashed) for every block; "
            f"(b,d) signed prediction minus reference. Plotted positive Matsubara window: "
            f"{omega[0]:.6g} to {omega[-1]:.6g}, {len(omega)} entries. "
            f"Unit convention: {unit}. Blocks are not independent physical samples. "
            "No curve alignment or oracle offset removal is applied."
        )
        if context:
            caption += " " + context
        finish(fig, axes, path, caption)
        plt.close(fig)
        fig, axes = plt.subplots(
            len(labels),
            2,
            figsize=(7.1, max(2.4, 1.35 * len(labels))),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        if context:
            fig.suptitle(context, fontsize=9)
        for b, label in enumerate(labels):
            for row, component in enumerate((np.real, np.imag)):
                ax = axes[b, row]
                for orbital in range(reference.shape[1]):
                    color = COLORS[orbital % len(COLORS)]
                    ax.plot(
                        omega,
                        component(reference[b, orbital, orbital]) * scale,
                        color=color,
                        label="reference" if orbital == 0 else None,
                    )
                    ax.plot(
                        omega,
                        component(predicted[b, orbital, orbital]) * scale,
                        "--",
                        color=color,
                        label="prediction" if orbital == 0 else None,
                    )
                ax.set_ylabel(("Re" if row == 0 else "Im") + f" ({native})")
                ax.set_title(label, fontsize=8)
        axes[0, 1].legend(frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\omega_n$" + f" ({native})")
        finish(
            fig,
            axes,
            path.with_name(path.stem + "_blocks.png"),
            "Block-resolved companion. Rows follow stored block labels; columns show Re and Im. "
            + caption,
        )
        plt.close(fig)


def trajectory_figure(path: Path, curves: list, final) -> None:
    """Resolve trajectories by block and numeric iteration color; overlay final."""
    if not curves:
        return
    with figure_style() as plt:
        from matplotlib import colors

        norm = colors.Normalize(min(n for n, _ in curves), max(n for n, _ in curves))
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(
            len(final.blocks),
            2,
            figsize=(7.1, max(2.5, 1.4 * len(final.blocks))),
            sharex=True,
            squeeze=False,
            layout="constrained",
        )
        for b, label in enumerate(final.blocks):
            for column, component in enumerate((np.real, np.imag)):
                ax = axes[b, column]
                for number, curve in curves:
                    for orbital in range(curve.values.shape[1]):
                        ax.plot(
                            curve.mesh.imag,
                            component(curve.values[b, orbital, orbital]),
                            color=cmap(norm(number)),
                            linewidth=0.8,
                        )
                for orbital in range(final.values.shape[1]):
                    ax.plot(
                        final.mesh.imag,
                        component(final.values[b, orbital, orbital]),
                        "k--",
                        linewidth=1.4,
                    )
                ax.set_title(label, fontsize=8)
                ax.set_ylabel(
                    ("Re" if column == 0 else "Im") + r" $\Sigma$ (native units)"
                )
        for ax in axes[-1]:
            ax.set_xlabel(r"$\omega_n$ (native units)")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            label="Stored iteration",
            fraction=0.025,
            pad=0.03,
        )
        finish(
            fig,
            axes,
            path,
            "Each row resolves one block's diagonal Sigma; color denotes actual "
            "stored iteration, black dashed denotes the final payload. No spin/orbital averaging. "
            "No convergence or phase classification is inferred. Units remain native.",
        )
        plt.close(fig)

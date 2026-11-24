"""Create HDF5 storage-consistency diagnostics, not convergence classifications."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dmft_data.run_diagnostics import classify_hdf5_runs


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("laptop-results/run_diagnostics"),
    )
    parser.add_argument(
        "--low-frequency-count",
        type=int,
        default=10,
        help=(
            "Number of lowest positive Matsubara entries used for relative RMS "
            "reporting (default: 10)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    summary = classify_hdf5_runs(
        args.source.resolve(),
        args.output.resolve(),
        low_frequency_count=args.low_frequency_count,
    )
    LOGGER.info(
        "Classified %d HDF5 runs; verified_mesh_files=%d",
        summary["file_count"],
        summary["files_with_verified_mesh"],
    )


if __name__ == "__main__":
    main()

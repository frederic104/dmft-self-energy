"""Create run-level HDF5 diagnostics without inventing physical conclusions."""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    summary = classify_hdf5_runs(args.source.resolve(), args.output.resolve())
    LOGGER.info(
        "Classified %d HDF5 runs; verified_mesh_files=%d",
        summary["file_count"],
        summary["files_with_verified_mesh"],
    )


if __name__ == "__main__":
    main()

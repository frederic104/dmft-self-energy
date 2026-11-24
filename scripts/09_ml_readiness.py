"""Write a fail-closed ML-readiness decision from a local dataset-build report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dmft_data.ml_readiness import write_ml_readiness_report


def main() -> None:
    """Evaluate the local dataset-build report without reading HDF5 payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-build-report",
        type=Path,
        required=True,
        help="Local JSON report written by scripts/04_build_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/local/ml_readiness.json"),
        help="Local path for the readiness decision.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    decision = write_ml_readiness_report(args.dataset_build_report, args.output)
    logging.getLogger(__name__).info("ML readiness status=%s", decision["status"])


if __name__ == "__main__":
    main()

"""Summarize a compatibility canonical HDF5 record population."""

import argparse
import logging
from pathlib import Path

from dmft_data.population import summarize_compatibility_canonical

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Write a machine-readable compatibility population summary."""
    parser = argparse.ArgumentParser(
        description="Summarize compatibility canonical HDF5 records"
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("data/processed/research_datasets/compatibility_deep_dmft/canonical.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/compatibility_population.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = summarize_compatibility_canonical(args.canonical, args.output)
    LOGGER.info(
        "Compatibility population status=%s; records=%d",
        report["status"],
        report["record_count"],
    )


if __name__ == "__main__":
    main()

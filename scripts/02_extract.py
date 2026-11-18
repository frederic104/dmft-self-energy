"""Entry point for representative DMFT HDF5 extraction."""

import logging
from pathlib import Path

from dmft_data.extraction import run_extraction

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run normalized extraction using the repository dataset configuration."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract configured HDF5 files")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs/dataset.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repository_root = Path(__file__).resolve().parent.parent
    report = run_extraction(args.config, repository_root)
    LOGGER.info(
        "Extraction status=%s; files=%d; failures=%d",
        report["status"],
        len(report["files"]),
        len(report["failures"]),
    )

if __name__ == "__main__":
    main()

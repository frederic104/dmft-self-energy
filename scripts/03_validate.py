"""Entry point for numerical extracted-data validation."""

import logging
from pathlib import Path

from dmft_data.validation import run_validation

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run validation using repository configuration."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate extracted DMFT data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "configs/validation.yaml",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    repository_root = Path(__file__).resolve().parent.parent
    report = run_validation(args.config, repository_root)
    LOGGER.info(
        "Validation status=%s; files=%d; failures=%d",
        report["status"],
        len(report["files"]),
        len(report["failures"]),
    )


if __name__ == "__main__":
    main()

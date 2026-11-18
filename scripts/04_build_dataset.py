"""Entry point for verified research-dataset construction."""

import logging
from pathlib import Path

from dmft_data.dataset import build_research_datasets

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build A/B datasets, auxiliary data, derived windows, and blocked splits."""
    import argparse

    parser = argparse.ArgumentParser(description="Build verified DMFT research datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs/dataset.yaml",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    repository_root = Path(__file__).resolve().parent.parent
    report = build_research_datasets(args.config, repository_root)
    LOGGER.info(
        "Dataset status=%s; independent_states=%d; warnings=%d",
        report["status"],
        report["blocked_splits"]["independent_physical_state_count"],
        len(report["warnings"]),
    )


if __name__ == "__main__":
    main()

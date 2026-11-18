"""Entry point for representative DMFT HDF5 extraction."""

import logging
from pathlib import Path

from dmft_data.extraction import read_hdf5_file

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Read a file supplied as the first positional argument."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract a representative HDF5 file")
    parser.add_argument("hdf5_file", type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    args = parser.parse_args()

    result = read_hdf5_file(args.hdf5_file, args.source_root)
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Extracted %d arrays from %s", len(result.arrays), args.hdf5_file)




if __name__ == "__main__":
    main()

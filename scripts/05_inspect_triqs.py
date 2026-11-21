"""Inspect TRIQS-style HDF5 files on a Linux machine.

Examples:
  python scripts/05_inspect_triqs.py --source /path/to/file.h5
  python scripts/05_inspect_triqs.py --source /path/to/hdf5 --output reports/triqs_inspection.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dmft_data.triqs_inspection import inspect_triqs_file


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="One HDF5 file or a directory containing HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/triqs_inspection.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    source = args.source.resolve()
    if source.is_file():
        files = [source]
        source_root = source.parent
    elif source.is_dir():
        files = sorted(
            (
                path
                for path in source.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".h5", ".hdf5"}
            ),
            key=lambda path: path.as_posix(),
        )
        source_root = source
    else:
        raise FileNotFoundError(f"Source does not exist: {source}")

    reports = [
        inspect_triqs_file(path, source_root)
        for path in files
    ]
    aggregate = {
        "status": "passed" if all(row["status"] == "success" for row in reports) else "failed",
        "source": str(source),
        "files": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Inspected %d HDF5 file(s); status=%s; report=%s",
        len(reports),
        aggregate["status"],
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

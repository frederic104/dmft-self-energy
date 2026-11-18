"""Entry point for source HDF5 inventory."""

from pathlib import Path

from dmft_data.inventory import run_inventory


def main() -> None:
    """Run inventory using the repository dataset configuration."""
    repository_root = Path(__file__).resolve().parent.parent
    run_inventory(
        repository_root / "configs" / "dataset.yaml",
        repository_root=repository_root,
    )


if __name__ == "__main__":
    main()

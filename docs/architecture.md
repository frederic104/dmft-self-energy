# Project Architecture

The repository is organized around a non-destructive, staged data workflow:

- `src/dmft_data/` contains reusable inventory, extraction, validation,
  provenance, metadata, convergence, and dataset-construction code.
- `scripts/` contains thin command-line entry points for the staged workflow.
- `configs/` contains YAML configuration for dataset paths and validation
  policy.
- `tests/` contains focused unit tests for package behavior.
- `reports/` stores machine-readable extraction and validation reports.
- `data/raw/` is reserved for original local data and is never modified.
- `data/processed/` is reserved for derived numerical arrays and datasets.

The initial repository only provides placeholders. It does not extract HDF5
files or create scientific datasets.

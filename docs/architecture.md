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

Phase 3 provides a strict representative HDF5 reader. It reconstructs
candidate G0, G, Weiss, hybridization, and Sigma arrays only from verified
native or real/imaginary representations. It preserves source paths,
attributes, iteration identifiers, channel labels, and separately verified
Matsubara meshes. Unsupported schemas raise explicit errors.

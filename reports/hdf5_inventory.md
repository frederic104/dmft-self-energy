# DFT+DMFT HDF5 Inventory

This inventory records HDF5 structure and provenance only.
It does not extract scientific arrays or infer physical meanings.

## Summary

- Files discovered: 1
- Successfully opened: 0
- Failed: 1
- Groups: 0
- Datasets: 0
- Failures: 1

## Name-Based Dataset Candidates

- No name-based candidates identified.

## File Status

| Relative source path | Status | SHA-256 | Error |
|---|---|---|---|
| `data/raw/hdf5` | failed | `` | Configured raw data directory does not exist: /workspaces/dmft-self-energy/frederic104/dmft-self-energy/data/raw/hdf5 |

## Unresolved Schema Questions

- Dataset names are candidates only; physical meanings, channels, and iterations must be established from schema-specific metadata.
- Convergence status is unknown unless explicitly represented in file or dataset attributes and independently validated.
- Complex representations, frequency conventions, and spin/orbital labels cannot be inferred by inventory inspection alone.

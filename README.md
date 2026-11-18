
# DMFT Self-Energy

## Research Objective

This project investigates physics-constrained latent
representation learning for electronic self-energy prediction
in strongly correlated materials.

The objective is to develop a computationally efficient
surrogate that complements DFT+DMFT calculations.

## Current Scope

The initial development focuses on Track A:

- Public DFT+DMFT dataset inspection
- HDF5 data extraction
- Physical and numerical validation
- Data provenance and metadata management
- Construction of verified training datasets

Track B (finite-bath Anderson Impurity Model ED)
will be developed separately.

## Data Sources

The initial dataset is obtained from the public
Zenodo DMFT-ML dataset.

Original data files are stored locally and are not
included in this repository.

## Data Processing Principles

1. Preserve all original data.
2. Distinguish physical states from calculation runs.
3. Preserve DMFT iteration and spin-orbital information.
4. Preserve complex-valued arrays and original frequency meshes.
5. Do not assume the last iteration is converged.
6. Record missing metadata as unknown.
7. Separate extraction validation from physical convergence.

## Development Status

- [ ] Repository structure
- [ ] HDF5 inventory
- [ ] HDF5 reader
- [ ] Data extraction
- [ ] Numerical validation
- [ ] Convergence assessment
- [ ] Training dataset construction

## Installation

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -e ".[dev]"
```

## Data Processing

The local data directory is configured through
`configs/dataset.yaml`.

Run the pipeline in the following order:

```bash
python scripts/01_inventory.py
python scripts/02_extract.py
python scripts/03_validate.py
python scripts/04_build_dataset.py
```

The commands will be enabled as the corresponding
modules are implemented.

## Reproducibility

All extraction settings, source checksums,
validation results, and dataset split definitions
should be preserved.

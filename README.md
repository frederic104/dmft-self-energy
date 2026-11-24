
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

- HDF5 extraction and storage diagnostics support the observed TRIQS schema.
- Paired preparation and CPU baseline training are available through one CLI.
- Physical input semantics and per-run convergence require external evidence.
- Causal learned decoders, controlled ED, and spectral validation remain future work.

## Installation

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -e ".[dev]"
```

For baseline experiments, install `pip install -e ".[dev,ml]"`.

## Recommended learning workflow

On the Ubuntu machine, activate `physai` and work in
`/home/frederic104/projects/dmft-self-energy`:

```bash
python -m dmft_data catalog --source data/raw/hdf5 --output data/metadata/run_review.json
# Inspect the catalog and set input/target paths in configs/experiment.yaml.
python -m dmft_data prepare --config configs/experiment.yaml
python -m dmft_data summary --config configs/experiment.yaml
python -m dmft_data train --config configs/experiment.yaml
```

The example runs an explicitly exploratory stored-array experiment, even while
convergence remains `not_assessed`. It keeps spin/orbital blocks together and
groups replicas before fitting. Primary experiments use reviewed physical-state
IDs, scoped convergence evidence, and verified input semantics. See
[the workflow and evaluation protocol](docs/learning_workflow.md) for review
format, grouping, model selection, output artifacts, and limitations.

## Data Processing

The local data directory is configured through
`configs/dataset.yaml`.

The earlier normalized-HDF5/compatibility export workflow remains available:

```bash
python scripts/01_inventory.py
python scripts/02_extract.py
python scripts/03_validate.py
python scripts/04_build_dataset.py
```

For a local laptop with large or restricted HDF5 data, create a small transfer
bundle without sending the source payload:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/06_collect_laptop_results.py \
  --source /path/to/local/hdf5 \
  --output laptop-results
```

Send the generated `laptop-results/laptop_collection.zip` back to the
workspace. The ZIP contains inventory, extraction, numerical-validation, and
TRIQS-style diagnostic reports; it does not contain original HDF5 payloads.
The collector replaces the supplied local source-root prefix with
`<LOCAL_SOURCE_ROOT>`, but review the bundle before sharing because HDF5
attributes may contain other project-specific information.

After local normalized-data processing and `04_build_dataset.py`, record a
fail-closed ML admission decision:

```bash
python scripts/09_ml_readiness.py \
  --dataset-build-report reports/research_dataset_build.json \
  --output reports/local/ml_readiness.json
```

`ready` only means the dataset-build report recorded explicitly verified
converged records and a non-overlapping material-blocked holdout. It is not a
claim that this script independently validated the HDF5 data, convergence,
causality, or spectral reconstruction.

It is not required before the paired baseline workflow above. Legacy convergence
registries now require per-run SHA-256, exact `verified_source_paths`, a
`verified_converged` status, and an evidence reference. Collection-wide booleans
and unreviewed HDF5 flags cannot certify primary targets.

### HDF5 storage diagnostics

`scripts/07_classify_hdf5_runs.py` compares final `G_iw` and `Sigma_iw`
blocks with every stored iteration. It reports storage equality and numerical
differences, including a configurable low-positive-Matsubara-frequency RMS
(`--low-frequency-count`, default 10). It is a storage/consistency diagnostic:
it does not establish DMFT convergence, and all outputs retain
`convergence_status: "not_assessed"` and `primary_target_candidate: false`.
Paths under `beta-variation` are only marked `parameter_variation_candidate`
pending metadata review. Warmup settings, iteration records, and spin/orbital
blocks are not independent physical states.

For a fail-closed raw-data provenance audit and target/split registry, see
[research-readiness audit](docs/research_audit.md). This creates local JSON
reports only; without independently supplied convergence evidence it selects
no primary ML targets.

## Reproducibility

All extraction settings, source checksums,
validation results, and dataset split definitions
should be preserved.

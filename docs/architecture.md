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

The paired learning path is now the recommended experiment interface:

```text
python -m dmft_data catalog -> local run_review.json
python -m dmft_data prepare -> paired arrays + provenance + replica summary
python -m dmft_data train   -> blocked split + train-only preprocessing/models
                            -> validation selection -> final test + checkpoints
```

`hdf.py` centralizes physical block filtering and selected quantity access,
reusing complex reconstruction and mesh validation in `extraction.py`.
`run_diagnostics.py`, `triqs_inspection.py`, and `research_audit.py` share these
helpers and the checksum implementation in `provenance.py`.
`learning/data.py` handles catalog/preparation, `learning/splits.py` owns replica
grouping and three-way partitions, `learning/models.py` owns fitted predictors,
and `learning/experiment.py` owns selection, evaluation and reproducible artifacts.

Numbered scripts remain compatibility/reporting interfaces. `04_build_dataset.py`
exports the earlier normalized canonical schema; `09_ml_readiness.py` assesses
that export's evidence report. Neither is a mandatory admission gate for an
explicit exploratory paired experiment. The newer command preserves the task
label through preparation, training, checkpoints and reported results.

Material/beta candidates are no longer labeled explicit physical states during
normalization. Existing processed files carrying that old derived flag must be
regenerated before using the older export path. The new workflow reads selected
raw groups and therefore does not inherit stale normalized IDs.

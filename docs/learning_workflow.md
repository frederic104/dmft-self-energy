# Paired Matsubara baseline workflow

The supported learning entry point is `python -m dmft_data`. It reuses the
existing complex reconstruction code and reads **selected quantity groups**
from local raw HDF5 files. It does not require creating the older compatibility
canonical HDF5 files first. Raw data remain unchanged.

## Ubuntu: install, inspect, prepare, train

The catalog is a local review template: block paths, original shapes and dtypes,
and beta with HDF5 source paths. It reads metadata only and prints a short count.
An existing review file is preserved; choose another output to repeat inspection.
Show a small selection of quantity roots from the saved file:

```bash
python - <<'PY'
import json
from pathlib import Path
runs = json.loads(Path("data/metadata/run_review.json").read_text())["runs"]
roots = {path.rsplit("/", 1)[0] for r in runs for path in r["quantity_blocks"]}
print("runs:", len(runs))
print("G0 roots:", sorted(p for p in roots if "/G0" in p)[:20])
print("final Sigma available:", "/dmft_output/Sigma_iw" in roots)
PY
```

Review `configs/experiment.yaml` before preparing. Its input selector
`/dmft_output/G0-1` is an **explicit proposed stored-array selection**: the name
does not verify DFT-derived G0 or Weiss semantics. Change it to an observed path
if needed. Target `/dmft_output/Sigma_iw` intentionally retains the archive's
final arrays, including BVO's final=iteration-5 behavior. Choosing
`/dmft_output/Sigma-15` defines a different exploratory experiment; use new output
directories and keep that choice in the saved config.

The public preprocessing pairs `G0-i` with `Sigma-i`, unlike this default.
See [source evidence and the next experiment](upstream_data_contract.md) for
the checked notebook reference, unresolved generation semantics, and a bounded
same-index comparison. The `summary` output reports actual prepared quantity
pairs, including per-run selector overrides, without reading raw arrays again.

```bash
python -m dmft_data prepare --config configs/experiment.yaml
python -m dmft_data summary --config configs/experiment.yaml
python -m dmft_data train --config configs/experiment.yaml
python -m dmft_data summary --config configs/experiment.yaml
```

Preparing explicitly reads the selected raw arrays on the user's Ubuntu machine.
No full raw-data processing is needed to run the synthetic tests:

```bash
pytest tests/test_learning.py tests/test_extraction.py \
  tests/test_run_diagnostics.py tests/test_research_audit.py -q
```

## Experimental unit and evidence

One prepared sample contains one run's **all spin/orbital blocks**, with real and
imaginary parts preserved. Stored iterations are selected by path, never made
into extra independent samples. `nwarmup` is retained as metadata where supplied
and never added as a physical model feature by default.

In `task: exploratory`, `grouping: material_beta_candidate` puts all runs with the
same directory material and observed beta in a conservative split group. This
permits testing the stored-array mapping with the present 62 files, while
physical states and convergence remain unresolved. It does not certify that
members have identical U, J, filling, structure, or basis. A supplied
`replica_group_id` can combine more runs when a more conservative block is needed.
`physical_state_id`, when explicitly supplied, takes precedence.

Training uses arithmetic run means within each group, giving each group equal
weight. Channels are never averaged. `replicas.json` reports run counts, exact
target duplicates, and RMS spread around each mean. That spread describes these
stored runs; it is not a Monte Carlo standard error or an effective sample size.
Within a group, meshes and supplied conditioning parameters must agree. Different
basis layouts need separate experiments or an explicit externally reviewed map.
Within a group the selected input and target paths and their declared semantics
must agree; differently staged arrays are not averaged. Different input or target
semantics require separate experiments. Training checks that selection and
frequency count still agree with the configuration used to prepare the data;
changing them requires preparing a new dataset directory.

For `task: primary`, add per-run reviewed fields in `run_review.json`:

```json
{
  "physical_state_id": "a-reviewed-state-id",
  "metadata_evidence": "reviewed input/log/publication reference",
  "input_semantics": "dft_derived_g0",
  "target_semantics": "impurity_self_energy",
  "convergence": {
    "status": "verified_converged",
    "evidence_source": "reviewed criterion, threshold, residual and log reference",
    "source_sha256": "SHA256_OF_THIS_RAW_FILE",
    "target_path": "/dmft_output/Sigma_iw"
  }
}
```

These fields are researcher-supplied evidence, not an automatic convergence
assessment. A checksum and exact target path bind the declaration to that file
and selection. A status string alone cannot promote any array. Input semantics
may instead be `dmft_weiss` or `aim_hybridization`; those define different
scientific tasks and must be run separately. Fixed U/J/filling/basis conventions
must be documented in metadata evidence and `assumptions`; varying known scalar
parameters can be listed in `parameter_features`. Missing values are not inferred.

## Splits, representation and evaluation

The YAML fixes validation and final test blocks before fitting. The example
holds out CVO for validation and BVO for testing; remaining materials train.
Other dimensions are `beta`, `parameter_block`, and `group_id`. Every replica in a
group stays in one partition, and an exact paired input/target payload cannot
cross partitions. Temperature holdouts within one material should be described
as temperature interpolation/extrapolation, not cross-material generalization.

Only training groups fit input/output scalers, target PCA and predictors.
Mean, nearest neighbor, PCA+Ridge, and direct MLP use the same split and seed list.
PCA compresses the self-energy, then Ridge predicts its coefficients from the
input. Requested latent dimensions 4/8/16 are capped at `n_train_groups - 1`;
the actual dimension is recorded. When only a few groups exist these requests
may become the same model. Validation reconstruction-only RMSE separates target
compression error from mapping error. MLP stopping uses training data, without
an internal random split of replicas. Optimizer warnings are saved.

Validation mean complex RMSE selects the specification; test scores are produced
only for that specification, across the preset seeds. The test set is not used
for hyperparameters, PCA, preprocessing, or stopping. To compare all models on
a final test, freeze the comparison protocol beforehand; do not use repeated
test-driven edits as model selection. Seed variation measures optimizer
sensitivity, not physical-state uncertainty.

The feature order is flattened input real/imaginary values, positive Matsubara
frequencies, then optional scalar parameters. There is no interpolation.
Different beta values retain their own grids and enter as explicit frequency
features; arrays at the same index are not assumed to have the same energy.
All samples must share the selected number of positive frequencies and channel
layout. The selection count is configurable and insufficient meshes are excluded
with reasons. Prepared manifests list every included run and every exclusion.

Metrics include complex/imaginary RMSE, MAE, maximum absolute error, relative RMS
with a recorded floor, low-frequency RMSE, upper-selected-window RMSE, and each
test group's error. The upper-window metric is **not an asymptotic moment test**.
The eigenvalue test of `(Sigma - Sigma†)/(2i)` checks the necessary negative
semidefinite sign on positive imaginary frequencies; it alone does not establish
causality. Reference and predicted sign violations are both reported.

`metrics.json`, `splits.json`, `feature_layout.json`, saved YAML-equivalent config,
dependency versions, git revision, data hashes, fitted scalers/PCA/model checkpoints
and test predictions with meshes are written to local output directories. Each
experiment/preparation uses a new directory so stale artifacts cannot be reused.
Checkpoints use joblib; load only locally generated, trusted files.

## Scope and remaining experiments

This implements the baseline stage. It does not yet implement a learned causal
decoder, IR/DLR comparison, ED validation, analytic continuation, filling/moment
tests, or a lattice DOS calculation. Their scientific prerequisites differ and
they should be added after establishing baseline performance and provenance.
Matsubara prediction accuracy cannot establish real-frequency spectral accuracy.

Mitra and Banerjee (2026), DOI
[10.1038/s41524-026-02277-6](https://doi.org/10.1038/s41524-026-02277-6), already
predict self-energy from non-interacting Green functions; that mapping alone is
not novelty. Their use of 15 iterations and warmup-conditioned curves is prior
methodology, not evidence that iterations/spin/orbitals are independent states.
The present study should test whether compact representations improve held-out
physical consistency relative to these baselines, after replica-aware evaluation.
The workflow also follows the training-only preprocessing principle in the
[scikit-learn guidance](https://scikit-learn.org/stable/common_pitfalls.html) and
keeps the DFT/Weiss distinction in the
[TRIQS DMFT formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html).


# DMFT Self-Energy

## Research Objective

This project investigates physics-constrained latent
representation learning for electronic self-energy prediction
in strongly correlated materials.

The objective is to develop a computationally efficient
surrogate that complements DFT+DMFT calculations.

## Research workflow

Use the smallest experiment that can answer a stated question and change the
next decision. For exploratory work, keep input and target meanings explicit,
check the ED limiting cases or Track A source evidence, keep state groups and
partitions separate, and compare simple baselines. Investigate extra diagnostics
when they explain a result; publication-scale checks are not required after every
small run. Preserve a new untouched test for the final frozen comparison.

The fixed-bath AIM is a controlled reference, not a real-material calculation.
The original [Mitra and Banerjee study](https://doi.org/10.1038/s41524-026-02277-6)
already predicts self-energy from non-interacting Green functions; that mapping
alone is not a novelty claim. [Arsenault et al.](https://doi.org/10.1103/PhysRevB.90.155136)
examined AIM Green-function representations and training-set size. A materials
study of [out-of-distribution evaluation](https://doi.org/10.1038/s43246-024-00731-w)
shows why coverage and the intended holdout matter; it does not make a random
split invalid for an explicitly labeled interpolation question. Reproducibility
guidance such as the [NeurIPS program report](https://www.jmlr.org/papers/v22/20-303.html)
motivates recording settings and variability. These sources inform the workflow,
not universal numerical acceptance thresholds or a claim of novelty.
For connecting ED to DMFT later, finite-bath accuracy and bath-size effects need
their own assessment, as in [Liebsch and Ishida](https://arxiv.org/abs/1109.0158).

## Current Scope

The initial development focuses on Track A:

- Public DFT+DMFT dataset inspection
- HDF5 data extraction
- Physical and numerical validation
- Data provenance and metadata management
- Construction of verified training datasets

Track B separately provides a controlled finite-bath Anderson impurity reference.

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
- Causal learned decoders and lattice spectral validation remain future work.
- Track B includes a small finite-temperature, two-bath AIM ED reference.

## Installation

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -e ".[dev]"
```

For baseline experiments, install `pip install -e ".[dev,ml]"`.

## Track B: finite-bath AIM reference

`dmft_data.ed` diagonalizes a spinful impurity and two spin-degenerate bath
sites in the 64-state Fock space. With orbital order
`(d_up, d_down, b0_up, b0_down, b1_up, b1_down)`, it diagonalizes `K=H-mu*N`:

```text
H = epsilon_d sum_sigma n_dsigma + U n_dup n_ddown
  + sum_l,sigma [epsilon_l n_lsigma + V_l (d_sigma^dagger c_lsigma + h.c.)]
G0(iw)^-1 = iw + mu - epsilon_d - sum_l V_l^2/(iw + mu - epsilon_l)
Sigma(iw) = G0(iw)^-1 - G_imp(iw)^-1
```

Use one common energy unit. `G_imp` is the thermal Lehmann Green's function,
spin averaged by default; no double-counting correction is applied. The
broadened `-Im G_imp(w+i eta)/pi` is an impurity spectral function. This
finite-bath reference has no DMFT self-consistency and is not a lattice DOS.

On Ubuntu, in `~/projects/dmft-self-energy`:

```bash
conda activate environment name
pip install -e ".[dev]"
pytest -q tests/test_ed.py
python -m dmft_data.ed --output data/processed/ed/aim_small.npz --samples 8 --seed 0
```

The local, Git-ignored NPZ contains the positive Matsubara mesh, parameters,
complex `green`, `g0`, `sigma`, and JSON metadata including the random seed and
NumPy version. Existing files are protected unless `--overwrite` is supplied.
The tests check analytic noninteracting and atomic limits, spectral
normalization, Matsubara causality and reproducibility. Finite bath size,
temperature and broadening limit the reference's physical scope.
The ED module also provides thermal impurity occupancy per spin. Its atomic-limit
checks use independent partition-function expressions; high-frequency tests use
the spin-symmetric moments `Sigma_inf = U*n_per_spin` and
`Sigma_1 = U^2*n_per_spin*(1-n_per_spin)`. These controlled AIM checks do not
establish the corresponding quantities or convergence in Track A data.
The generator varies U independently, while G0 does not contain U. A future
learning task must supply U (and other varying quantities not identifiable from
its input) or explicitly fix them; G0 alone does not define a unique Sigma
across independently varied U. The existing Track A blocked-partition helper can
be reused only after defining ED physical-state groups and parameter blocks;
the ED generator currently creates no train/validation/test split.

For an exploratory fixed-U CPU baseline, use a fresh local output path:

```bash
python -m dmft_data.ed --output data/processed/ed/fixed_u.npz --samples 60 --seed 0 --fixed-u 2.0 --beta 20
python -m dmft_data.ed_baselines --dataset data/processed/ed/fixed_u.npz --output reports/local/ed_fixed_u --seed 0
```

The second command holds out low and high impurity-level intervals for validation
and test, selects among fixed baseline settings on validation only, and saves the
selected model, split indices and metrics. This is an illustrative parameter
holdout, not evidence of generalization across arbitrary baths or materials.
Use fresh output paths for new experiments; do not retune on the reported test.

For exploratory comparisons, run the following from the repository root with
`physai` activated. These small example counts and ranges are starting points,
not a final assessment; keep a separate final test dataset for later research.
The same dataset is used for the random and epsilon splits, but their train and
test states differ, so their scores compare protocols rather than matched cases.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# A: independent-state interpolation versus epsilon-blocked extrapolation
python -m dmft_data.ed --output data/processed/ed/compare_a.npz --samples 100 --seed 13 --fixed-u 2 --beta 20
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_a.npz --output reports/local/aim_a_random --split random --seed 13
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_a.npz --output reports/local/aim_a_blocked --split epsilon --seed 13
# B: repeat the split comparison on a fixed bath; change the bath for another case
python -m dmft_data.ed --output data/processed/ed/compare_b.npz --samples 100 --seed 13 --fixed-u 2 --beta 20 --fixed-bath -1 1 0.5 0.7
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_b.npz --output reports/local/aim_b_random --split random --seed 13
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_b.npz --output reports/local/aim_b_blocked --split epsilon --seed 13
# C: U varies; it is explicitly appended to G0 as an input feature.
python -m dmft_data.ed --output data/processed/ed/compare_c.npz --samples 100 --seed 13 --beta 20
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_c.npz --output reports/local/aim_c_u --split u --seed 13
python -m dmft_data.ed_plot --result reports/local/aim_c_u
```

`--split u` reserves U in [1.5,2) for validation and [2,2.5) for test;
remaining sampled U values are training. These are interior holdouts, not
outside-range extrapolation. The PNG shows individual test-state errors rather
than smoothing or averaging over baths, and the worst low-frequency test curve.
Each run saves predictions and reference values for detailed inspection. ED
parameters, fixed bath and sampling ranges are retained in the local dataset;
metrics include split indices, seed, model settings and fit warnings. All test
figures and metrics are exploratory once examined; do not use them as untouched
final evaluation. The chosen fixed bath and intervals are illustrative.

### Latent comparison (exploratory)

The fixed-bath interpolation run found nearest neighbor competitive, while the
blocked run was harder. This motivates comparing compression and a nonlinear
bottleneck against the same baselines before adding a causal decoder. It does
not establish a generalization or novelty claim.

Use a fresh dataset and output directory, for example:

```bash
python -m dmft_data.ed --output data/processed/ed/latent_example.npz --samples 100 --seed 31 --fixed-u 2 --beta 20 --fixed-bath -1 1 0.5 0.7
python -m dmft_data.ed_baselines --dataset data/processed/ed/latent_example.npz --output reports/local/aim_latent_random --split random --seed 31 --include-latent
```

`--include-latent` compares PCA+Ridge and unconstrained deterministic MLPs
with 1, 2, 4, 8 and 16 bottleneck units alongside the mean, nearest and direct MLP.
The MLP uses ReLU layers of widths 64, latent dimension, 64 with linear output;
training-only standardization and a 500-iteration optimizer budget match the
direct MLP. PCA is target-side compression, not a learned input encoder. Its
dimensions are not silently clipped: at least 17 training states and 16 real
target features are required for dimension 16.
Each candidate's validation errors, sign-violation fraction, seed and fit
warnings are saved; only the validation winner is evaluated on test. The last
8-point metric is a finite-window error, not a high-frequency moment test.
The MLP decoder is unconstrained: sign checks on sampled Matsubara points do not
prove causality. Causal decoders and real-frequency property reconstruction
remain subsequent work, not part of this comparison. Preserve a separate final
test before making research claims; previously inspected tests are exploratory.

For a targeted training-stability question, use `--compare-activations` with
`--include-latent --validation-only` and repeat a prespecified set of
`--model-seed` values into separate directories. This adds matched tanh variants
of the direct and latent MLPs; tanh applies to **all hidden layers**, not only
the bottleneck. The validation-only option does not predict or score reserved
test samples, but it still reads the local dataset. Model initialization seeds
on the same physical states are not independent samples. The output records
activation, stopping settings, warnings and train/validation bottleneck behavior;
tanh saturation is reported instead of calling negative tanh activations dead.

```bash
for s in 31 32 33; do
  python -m dmft_data.ed_baselines --dataset data/processed/ed/latent_example.npz \
    --output "reports/local/activation_seed_${s}" --split epsilon --seed 31 \
    --model-seed "$s" --include-latent --compare-activations --validation-only || break
done
```

For the fixed-bath question, `--include-epsilon-linear` adds a baseline using
**epsilon_d directly**. The other candidates use G0, so this is a comparison
with different input information. It requires fixed U, bath parameters, mu and
beta across the dataset and at least two distinct training epsilon values;
duplicate training epsilon values are rejected. It fits only training targets,
linearly interpolates each real and imaginary component, and extrapolates with
the slope of the nearest endpoint pair. The latter is a deliberately simple,
potentially unstable baseline for the blocked split, not a physical guarantee.
Metrics record the training range and validation counts below, within and above
that range; test coverage is recorded only when test evaluation is enabled.

```bash
python -m dmft_data.ed_baselines --dataset data/processed/ed/latent_example.npz \
  --output reports/local/aim_linear_validation --split epsilon --seed 31 \
  --model-seed 31 --include-latent --include-epsilon-linear --validation-only
```

The reserved high-epsilon block from an already inspected dataset remains
exploratory. Before any final evaluation, freeze the candidate selection,
training and evaluation protocol and reserve independently generated physical
states. A saved validation winner can be evaluated once with:

```bash
python -m dmft_data.ed_external_test \
  --development data/processed/ed/latent_example.npz \
  --result reports/local/aim_linear_validation \
  --test data/processed/ed/independent_final.npz \
  --output reports/local/independent_final_evaluation
```

Use only trusted local joblib artifacts. The evaluator verifies the development
dataset hash, fixed parameters, exact frequency mesh and distinct physical
states; it cannot establish that a dataset was never previously examined. It
uses the saved validation winner without refitting and reports errors separately
below, within, and above its training epsilon range. Only the selected model is
saved by the validation workflow, so this does not compare every candidate on the
external set. Freeze the protocol before using a genuinely independent test.
No interpolation or extrapolation result establishes lattice DOS accuracy or
causality, and extrapolated values need not obey the Matsubara sign constraint.


Inspect the per-candidate validation errors, train loss and bottleneck metrics;
keep all seeds and identify optimization failures. The same 500-iteration limit
does not equalize wall-clock work or model capacity. The old default also scores
the exploratory test; use `--validation-only` while developing, and freeze a
protocol with a fresh independent test before a final claim. Current sign checks
are sampled necessary conditions, not proof of causality; spectral properties
and lattice DOS remain later, separate analyses.

The fixed-bath example is a one-parameter family, but a one-dimensional nonlinear
curve need not be well represented by one linear PCA component. The exploratory
blocked run showed a real offset and low-frequency sign violations; therefore
`diagnostics.json` records per-test-state oracle real offsets, shape residuals and
individual positive-imaginary points. Offsets use true targets and are not a
deployable correction. Training-only PCA explained variance and reconstruction
errors on train and validation are diagnostic, not model selection scores.

Use `--model-seed` to vary optimizer initialization while holding `--seed` and
the random partition fixed; both are recorded in `metrics.json`. With an epsilon
blocked split the thresholds are fixed, and changing split seed does not change
its partition. Repeat runs into fresh directories, retaining seed-level results;
the same maximum MLP iteration count does not imply equal computational budgets
or equal tuning. Validation and test data must never be used for fitting scalers
or PCA. These diagnostics do not prove causality or establish the self-energy
high-frequency moment, and they do not assess lattice DOS.

`diagnostics.json` also records each MLP's training loss curve and convergence
settings. The loss is scikit-learn's regularized objective on standardized
targets, not the complex RMSE in native units; iteration counts alone do not
identify why optimization stopped. For each latent MLP, the train and validation
summaries count bottleneck units that were never positive, samples with all-zero
bottlenecks, and activation variability. "Never positive" is an observation on
the specified samples, not proof that a unit can never activate. These diagnostics
do not use test inputs or targets and do not establish physical meaning of a
latent coordinate. Compare parameter counts as well as bottleneck dimensions:
a narrow representation does not imply fewer trainable parameters. Model seeds
on one fixed split are repeated initializations, not independent physical states.

## Recommended learning workflow

On the Ubuntu machine, activate `environment` and work in
`~/projects/dmft-self-energy`:

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

After training, `python -m dmft_data diagnose --config configs/experiment.yaml`
bundles physics definitions, signed block errors and plots without retraining.
See [physics definitions and diagnostics](docs/physics_diagnostics.md) for the
review template and optional, explicitly selected raw-run/Dyson checks.
The [physics validation protocol](docs/physics_validation_protocol.md) sets out
the next reference, convergence and observable checks. Diagnostics include
block-resolved PNG/PDF figures, caption drafts and a machine-readable check plan.

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

`python -m dmft_data storage --source data/raw/hdf5 --output reports/local/run_diagnostics`
compares final `G_iw` and `Sigma_iw`
blocks with every stored iteration. It reports storage equality and numerical
differences, including a configurable low-positive-Matsubara-frequency RMS
(`--low-frequency-count`, default 10). It is a storage/consistency diagnostic:
it does not establish DMFT convergence, and all outputs retain
`convergence_status: "not_assessed"` and `primary_target_candidate: false`.
Paths under `beta-variation` are only marked `parameter_variation_candidate`
pending metadata review. Warmup settings, iteration records, and spin/orbital
blocks are not independent physical states.
The older `scripts/07_classify_hdf5_runs.py` command remains available and uses
the same implementation. An absent source directory fails clearly; an empty
directory produces a warning rather than silently looking like a useful scan.

For a fail-closed raw-data provenance audit and target/split registry, see
[research-readiness audit](docs/research_audit.md). This creates local JSON
reports only; without independently supplied convergence evidence it selects
no primary ML targets.

## Reproducibility

All extraction settings, source checksums,
validation results, and dataset split definitions
should be preserved.

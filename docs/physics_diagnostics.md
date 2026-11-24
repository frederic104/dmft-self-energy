# Physics definitions and diagnostic bundle

The experiment's numeric errors do not establish acceptable DMFT accuracy.
Record definitions and their sources, then compare the same quantities, basis,
energy conventions and solver stages. No universal meV tolerance is assumed.

## Ubuntu workflow

```bash
conda activate physai
cd /home/frederic104/projects/dmft-self-energy
python -m pip install -e ".[dev,ml]"

# This copies an unresolved template, not a physical-state or convergence certificate.
# Skip the copy if you already have a manually reviewed file.
cp -n configs/physics_review.example.yaml data/metadata/physics_review.yaml

# Read saved artifacts only; no retraining or raw HDF5 traversal.
python -m dmft_data diagnose --config configs/experiment.yaml \
  --definitions data/metadata/physics_review.yaml
```

Outputs are under the existing experiment's `physics_diagnostics/`:

- `summary.md`: definition/evidence table, nearest group, errors and limitations.
- `diagnostics.json`: machine-readable evidence, unit decision, signed offsets,
  per-block errors, selected neighbor identities, and requested raw-run checks.
- `block_metrics.csv`: one row per seed/test-group/block; metrics stay in native
  units, and signed offsets retain every matrix element.
- `seed_*_group_*.png`: Re/Im reference versus prediction, with signed errors on
  the actual positive Matsubara frequency axis. The plotted curves are diagonal
  elements; numeric errors include all matrix elements.

Definition entries have `value`, `status` and `evidence`. Only `status: reviewed`
with a nonempty evidence reference and a value counts as reviewed. Record the
actual publication/input/log section or HDF5 path. Defaults apply to all runs;
`per_run` maps exact source-relative filenames to overrides of definition fields.
Missing fields remain unresolved. Statements are researcher attestations; the
program does not verify that the cited input/log proves the statement.

`energy_unit` describes the common native unit of Sigma **and omega**. Mark eV
reviewed only after checking the source convention; then plots convert both axes
to meV. With unknown/mixed units no conversion is made. Review `beta_unit`
separately as its reciprocal. G and Sigma do not have the same dimensional units.
CSV/JSON metrics always remain native and explicitly say so.

For a fresh review of an existing experiment, use a new output directory:

```bash
python -m dmft_data diagnose --config configs/experiment.yaml \
  --definitions data/metadata/physics_review.yaml \
  --output reports/local/experiments/exploratory_v1/physics_review_2
```

## Target and iteration lineage

Read a raw run only by explicitly selecting its source-relative filename:

```bash
python -m dmft_data diagnose --config configs/experiment.yaml \
  --definitions data/metadata/physics_review.yaml \
  --output reports/local/experiments/exploratory_v1/physics_BVO \
  --raw-run BVO/BVO_beta40_nwarmup100000.h5
```

This verifies the raw checksum against preparation and compares final G/Sigma
with each stored iteration on identical block/mesh layouts. It records exact
matches and RMS changes to the previous stored iteration. A plot overlays mean
diagonal Sigma trajectories. The numeric report retains all blocks. Within-run
up/down differences are descriptive; paramagnetism, orbital degeneracy and
cross-material orbital correspondence are not assumed. Missing/malformed
quantities are recorded as raw-run failures, never silently treated as matches.

The default uses the prepared frequency count. `--raw-count 1025`, for example,
explicitly requests a wider positive-frequency range if present. These plots are
not tail fits. A mean over the first 64 frequencies is not Sigma-infinity, and
iteration plateaus do not certify physical convergence. Warmup variants are not
independent random seeds or Monte Carlo error bars.

## Optional, evidence-scoped Dyson consistency

The template includes a commented `dyson_checks` record. A check runs only for a
requested `--raw-run` and requires exact file checksum, explicit Weiss/G/Sigma
paths, correct quantity semantics, solver Sigma without DC subtraction, and
reviewed evidence that all arrays belong to the same solver/mixing stage and
basis. It evaluates the matrix residual

`inverse(Weiss) - inverse(G_imp) - Sigma_solver`.

It never applies this impurity identity to a projected DFT G0. Numerically equal
iteration labels alone are insufficient: pre/post mixing, symmetrization and
tail processing can differ. Singular inverses or incompatible layouts are
reported as failures. The residual describes algebraic consistency; it does not
establish a self-consistent lattice DMFT solution or convergence threshold.

## Reading the report

Keep the original predictive RMSE as the performance number. The frequency-mean
REAL error is a per-matrix-element descriptive offset; its sign is saved. The
oracle-centered residual uses the known target and cannot be deployed as an
improved predictor or interpreted directly as a Hartree/DC correction. No
gauge alignment is applied to training targets.

High-frequency moments, filling, Z, spectral functions and lattice DOS are
explicitly deferred. Their assessment needs additional reference information:
occupancy/interactions and reliable tails; an applicable metallic regime for Z;
Hamiltonian/projectors/DC and validated continuation for lattice spectral claims.
The necessary source conventions are described in the
[TRIQS DMFT formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html)
and [CTHYB moments guide](https://triqs.github.io/cthyb/latest/guide/high_freq_moments.html).
[VASP EDIFF](https://vasp.at/wiki/index.php/EDIFF) is a total-energy iteration
termination criterion and is not a self-energy prediction tolerance.


# Agent Guidelines

## Before Implementation

- Read README.md.
- Read .github/copilot-instructions.md.
- Inspect the current repository structure.
- Inspect existing tests.
- Do not assume that all HDF5 files share an identical schema.

## Implementation Strategy

Work incrementally.

Priority:

1. HDF5 inventory
2. Representative HDF5 reader
3. Complex array extraction
4. Numerical validation
5. Convergence assessment
6. Metadata normalization
7. Training dataset construction

Do not implement all stages in one change.

For Track B, progress incrementally from the verified finite-bath reference to
baselines and controlled representation comparisons. The HDF5 ordering above is
specific to Track A and does not block a separate AIM experiment.

## Decision-focused research workflow

- Before a study, state the question, the baseline, the independent physical
  unit, and the next decision the result can change. Prefer one minimal experiment
  to repeated full-data audits or a new report for every run.
- Always preserve provenance and independent splits, fit preprocessing on
  training data only, and check numerical validity. Match validation to the
  question: random grouped splits describe interpolation; blocked splits probe
  their specified holdout. Neither implies broader generalization by itself.
- During development use validation-only runs and a small prespecified set of
  initialization seeds. Record the split and every seed; seeds measure optimizer
  variation, not independent physical samples. Compare candidates on consistent
  splits and declared resource/tuning budgets, and retain failures and warnings.
- Investigate extra diagnostics when they can resolve a concrete uncertainty,
  such as a collapsed bottleneck. Do not repeatedly expand causality, spectral,
  convergence or publication checks as an automatic gate for exploratory code.
  Keep results and assumptions labeled; reserve stronger evidence for claims.
- Freeze the comparison protocol before any final test. Previously inspected test
  data used in development are exploratory. Evaluate only a fresh independent
  test after selection; do not invent a validated final result.
- Track A input semantics and per-run target convergence remain unresolved until
  supported by evidence. Track B finite-bath impurity results do not establish
  lattice DMFT convergence, real-material DOS, or spectral accuracy.

## Scientific Review

When a data field is ambiguous:

- Preserve the original field.
- Record its source path.
- Mark its physical interpretation as unresolved.
- Ask for clarification instead of guessing.

## Reporting

For every implementation task, report:

- Files changed
- Functions implemented
- Tests executed
- Test results
- Known limitations
- Unverified physical assumptions

## Local-data workflow

- Raw and processed HDF5 data stay on the local WSL2 laptop and are never
  committed, uploaded, or copied into this repository.
- Keep generated transfer bundles in `laptop-results/` or `reports/local/`.
  Before sharing a report, review it for local paths and attributes that are
  not needed for schema or validation review.
- Treat collection, extraction, and final-iteration equivalence as technical
  diagnostics only. They are not evidence that a DMFT run converged.
- Run `scripts/09_ml_readiness.py` after producing the local dataset-build
  report. A `not_ready` result blocks ML performance claims; it does not
  diagnose the missing scientific evidence.
- Work on a feature branch and do not merge or push to `main` automatically.

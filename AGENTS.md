
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

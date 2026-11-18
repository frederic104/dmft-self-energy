
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

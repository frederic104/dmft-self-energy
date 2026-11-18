
# Repository Instructions

## 1. Research Objective

This project investigates physics-constrained latent
representation learning for electronic self-energy prediction.

The current priority is data validation and extraction,
not model development.

The surrogate is intended to complement DFT+DMFT,
not replace it.

## 2. Scientific Requirements

- Distinguish physical states, calculation runs,
  DMFT iterations, spin channels, and orbital channels.
- Never treat iterations or spin/orbital channels
  as independent physical samples.
- Preserve complex-valued Green's functions and self-energies.
- Preserve original Matsubara frequency meshes.
- Do not assume the last stored iteration is converged.
- Do not infer missing U, J_H, filling, or other
  physical parameters.
- Distinguish DFT-derived G0, DMFT Weiss functions,
  and AIM hybridization functions.
- Keep impurity spectral functions distinct from lattice DOS.

## 3. Data Integrity

- Never modify original data files.
- Record source paths and checksums.
- Preserve source HDF5 dataset paths and attributes.
- Record extraction software versions and configuration.
- Never silently skip failed files.
- Store warnings and errors in machine-readable reports.
- Mark unknown metadata explicitly as unknown.

## 4. Extraction Requirements

- Use modular Python functions.
- Use h5py and NumPy for HDF5 extraction.
- Preserve real and imaginary components.
- Validate array dimensions before conversion.
- Preserve spin-orbital labels.
- Preserve iteration identifiers.
- Do not silently interpolate or truncate frequency meshes.
- Make extraction reproducible and idempotent.

## 5. Validation Requirements

Check:

- Dataset existence
- Array shape and dtype
- Finite numerical values
- Frequency mesh consistency
- Complex number reconstruction
- Duplicate or equivalent datasets
- Iteration-to-iteration differences
- Metadata consistency
- Convergence status

Extraction success does not imply physical convergence.

## 6. Dataset Construction

- Use verified converged DMFT results as primary targets.
- Keep intermediate iterations in auxiliary datasets.
- Do not create splits before identifying physical states.
- Use material-, temperature-, and parameter-blocked splits.
- Never claim physical generalization from random splits alone.

## 7. Development Workflow

Before changing code:

1. Inspect existing modules and tests.
2. Explain the proposed modification.
3. Implement a minimal modular change.
4. Add or update tests.
5. Run relevant tests.
6. Report actual execution results.

Never claim code was executed or validated
unless it was actually run.

Do not fabricate scientific results or metadata.

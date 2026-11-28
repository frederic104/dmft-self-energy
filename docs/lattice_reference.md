# Explicit lattice Dyson reference

This small NumPy module evaluates a supplied self-energy in a supplied
orthonormal Hamiltonian. It complements the scalar AIM routines in
`ed_properties.py`: the lattice operation requires k-resolved inversion before
projection/averaging and therefore cannot use an already averaged G0 in the
scalar AIM Dyson routine. It does not train a predictor, perform analytic
continuation, run DFT/DMFT, adjust filling, or infer material conventions.

## Reproduce the synthetic example

From an environment with the repository and its existing `ml` dependencies:

```bash
python scripts/lattice_reference_demo.py \
  --output reports/local/lattice_demo_v1
```

The example uses `H(k) = -2 cos(k)` and a chosen causal rational self-energy
`Sigma(z) = 0.4 + 0.3/(z - 1.5)`, with a declared DC potential of 0.4 and mu=0.
This DC choice is a synthetic static cancellation, not a material DC prescription.
The BZ uses 256 uniformly weighted points; the path uses 121 points from 0 to pi.
Both use 801 frequencies from -4 to 4 at eta=0.08 in model units. These are
illustration settings, not a demonstrated convergence study.

Outputs include `lattice_demo.png`, `demo.json`, input arrays/configs and three
subdirectories (`bz`, `path`, `noninteracting`) with `lattice.npz` and
`summary.json`. They are synthetic results, not model predictions or evidence
for SVO/BVO spectra. Existing output directories are refused to avoid overwrites.

## Normalized input contract

No raw HDF5 archive is accepted. An eventual converter must document the basis,
units, spin convention and source fields. Input NPZ files contain numerical
arrays and are loaded with `allow_pickle=False`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `hopping` | `(nk, nb, nb)` | Hermitian H(k), orthonormal band basis, common energy origin |
| `projectors` | `(nk, nc, nb)` | P(k) maps bands to the same local correlated basis in which Sigma/DC are stored |
| `sigma` | `(nw, nc, nc)` | Complex impurity self-energy **before** DC subtraction, on `z` |
| `dc` | `(nc, nc)` | Hermitian DC potential, never the scalar `dc_energ` functional |
| `mu` | scalar | Real chemical potential in H's energy convention |
| `z` | `(nw,)` | Frequencies relative to mu; ordered positive imaginary points or increasing `omega + i eta` with constant eta>0 |
| `weights` | `(nk,)` | BZ only: nonnegative quadrature weights summing to one |
| `k_distance` | `(nk,)` | Path only: strictly increasing path coordinate; no weights |

Use [the example YAML](../configs/lattice_reference.example.yaml). Explicit
fields declare orthonormal basis, before-DC Sigma, relative-to-mu frequency axis,
one explicit spin block, units (`eV` or `model_units`), interpretation
(`synthetic` or `exploratory_material`), sampling and frequency domain. These
declarations are not physical verification and never mark an archive ready.

```bash
python -m dmft_data.lattice \
  --input reports/local/lattice_demo_v1/bz_input.npz \
  --config configs/lattice_reference.example.yaml \
  --output reports/local/lattice_reference_check_v1
```

For every k and frequency the implementation computes

```text
Sigma_band(k,z) = P(k)^dagger [Sigma(z) - dc] P(k)
G(k,z) = [(z + mu) I - H(k) - Sigma_band(k,z)]^-1
G_projected(k,z) = P(k) G(k,z) P(k)^dagger
G_local(z) = sum_k weights(k) G_projected(k,z)         [BZ only]
A(k,omega) = -Im trace G(k,omega+i eta) / pi           [real-frequency only]
DOS(omega) = sum_k weights(k) A(k,omega)              [BZ only]
```

The output always contains `z`, `green_k` and `projected_green_k`. A real-frequency
run also has `spectral_k` and `projected_spectral_k`; BZ runs add `green_local`
and, on the real axis, `dos` and `projected_dos`. Path runs preserve `k_distance`
and produce neither DOS nor local BZ averages. Full band matrices are not
averaged across k because their gauge can differ. Projectors may be rectangular;
the projected high-frequency coefficient is the weighted `P P^dagger`, which
is not silently replaced by an identity matrix.

The trace DOS covers the retained band space, not automatically every band of
the material. There is no spin factor, symmetrization, clipping, weight
renormalization or extrapolation. A nonorthogonal atomic basis needs an overlap
metric and different handling; this module does not accept that convention.
The current fixed-size arrays describe one explicit spin block and one supplied
correlated subspace; general multi-shell/SOC archive conversion is not implemented.
Finite eta and a finite frequency window affect spectral weight. Do not
renormalize a finite-window DOS to hide missing weight.

## Verification and provenance

`tests/test_lattice.py` checks independent Lorentzian limits; DC subtraction
exactly once; noncommutativity of local averaging and Dyson inversion; invariance
under a common shift of H/mu; complex unitary basis covariance; the leading
high-frequency moment; a rational self-energy against an enlarged Hermitian
Hamiltonian resolvent; and rejection of mismatched conventions, non-Hermitian
DC, invalid weights and path/DOS confusion. These test the numerical kernel,
not material identification or convergence.

```bash
python -m pytest -q tests/test_lattice.py
```

Each run records input, config, output and implementation SHA256 hashes, Python
and NumPy versions, declared conventions and output shape. Shared summaries
omit local absolute roots and arbitrary config metadata. The negative spectral
trace count is a limited diagnostic: nonnegative traces do not certify a
positive semidefinite matrix spectrum or causal input Sigma. Metadata records
`material_validation: not_assessed` even for an exploratory material input.

The next material step requires the alignment evidence described in
[the integrated research review](research_integration.md). The current SVO
archive's DFT/DC payload presence is not a substitute for that evidence.

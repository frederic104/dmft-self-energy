# Integrating representation learning with spectral postprocessing

Review date: 2026-11-28. This is a literature-informed implementation decision,
not a new experimental result. The [closed AIM comparison](track_b_closeout.md)
and [upstream data contract](upstream_data_contract.md) remain authoritative for
what this repository has actually demonstrated.

## Decision and corrections to the proposed plan

Integrate complex self-energy learning, observable-aware evaluation and an
explicit lattice postprocessing boundary. Keep continuation and material
reconstruction conditional on evidence. Do not restart the closed candidate
search or recreate the existing HDF5 audit, scalar causal decoder and Sigma+G
objective under new names.

The essential prior work already learns self-energy from noninteracting Green
functions. Its abstract names **CaVO3**, not CrVO3, among the transfer materials.
The upstream repository README instead expands CVO as CrVO3. Preserve the raw
CVO label until archive or author evidence resolves that discrepancy; do not
silently rename files or assert a CrVO3 d1 example.
[Mitra and Banerjee (2026)](https://doi.org/10.1038/s41524-026-02277-6),
[Deep-DMFT repository](https://github.com/hrishitbanerjee/Deep-DMFT).

The upstream learning example emphasizes imaginary components on a reduced
frequency window. That is not a statement that its raw HDF5 archives lack
complex data. The nominated SVO file has already been inspected and contains
complex quantities and DFT/DC payloads. Presence, basis alignment and physical
convergence are separate questions. Neither every intermediate iteration nor
every low-warmup run can be declared unconverged solely from its identifier.
They remain auxiliary/exploratory until their role is supported. Warmup length
is not a measured noise variance. Keep all replicas of the same candidate
material/temperature group together under the existing split policy.

The proposed Zenodo identifier is
[10.5281/zenodo.17215588](https://doi.org/10.5281/zenodo.17215588). Its record and
reuse terms could not be independently retrieved during this review. Before
redistribution, record the actual archive version DOI, file checksum and
applicable data/code terms. A public repository and the paper's open-access
status do not establish identical reuse terms for every artifact. No archive
is redistributed by this change. Open-source DFT still has licenses and
computational costs; the practical constraint is no commercial-license fee.

## Physical and numerical boundaries

### Dyson consistency is conditional on which G0 is supplied

For an AIM, the Weiss field obeys

`Gimp^-1 = Gweiss^-1 - Sigma`,
`Gweiss^-1 = z + mu - epsilon_d - Delta(z)`.

A projected noninteracting lattice Green function is generally not the
interacting DMFT Weiss field. In the lattice, embed Sigma in each k-space
Hamiltonian, invert there, then project and average. Averaging and inversion
do not commute. DMFT additionally imposes `Gimp = Gloc` and updates the bath;
postprocessing a supplied Sigma does not perform that cycle. See the
[DFTTools formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html).

Consequently, a successful Track B OOD experiment cannot identify the meaning
of Track A `/dmft_output/G0-1`. The existing same-stage Dyson test and generator
evidence in [the validation protocol](physics_validation_protocol.md) address
that question. Even an algebraic match alone cannot establish how the source
quantity was generated or whether DMFT converged.

There is also an identifiability condition on the learning problem: the same
noninteracting AIM G0 can occur at different U and yield different Sigma.
If a future study varies interactions, condition the predictor on U/J and the
relevant temperature/mesh and subspace conventions, or explicitly restrict
those quantities. G0 alone cannot define a universal impurity solver across
unspecified interactions. The current fixed-U/fixed-beta experiment avoids
this particular ambiguity. Transferring its numerical methods to a material
pipeline is reasonable; transferring learned single-orbital finite-bath physics
to a multi-orbital, Hund-coupled self-consistent lattice problem is a separate
hypothesis that the present experiment has not tested.

With the same verified bath, the exact resolvent identity is
`Gpred - Gref = Gpred (Sigma_pred - Sigma_ref) Gref`.
It explains why a small Sigma error can produce a large G error near a sensitive
denominator. The existing Sigma+G objective is therefore a justified comparison,
but its weights and frequency scaling must be selected on validation data.
It guarantees neither correct occupation nor real-axis spectra. Occupation
also requires adequate frequency tails, normalization and a defined filling
protocol. The new lattice kernel keeps mu fixed; fixed-filling comparisons
would require a separately tested chemical-potential solve.

Call this **Dyson-coupled training**, not Delta learning. Delta-ML conventionally
learns the difference between a baseline and a target model; AIM hybridization
Delta is a third, distinct quantity.
[Ramakrishnan et al. (2015)](https://doi.org/10.1021/acs.jctc.5b00099).
The [real-offset counterfactual](aim_error_propagation.md) uses reference targets
and is not a deployable residual predictor. A future residual model would need
to predict its correction from inputs alone; the difference of two causal
self-energies need not itself have positive spectral weights.

### Causal representation is useful but insufficient

The current scalar `PoleBaseline` already represents a real constant plus
positive residues on real poles and can evaluate that rational function at
complex z. Its real-axis use is direct function evaluation, not a second
analytic-continuation algorithm applied to sampled predictions. The closed
comparison does not support adopting it as the primary predictor.

DLR supplies an efficient basis controlled by temperature, cutoff and numerical
tolerance. It is a compression option, not an automatic positivity constraint
or a unique spectral reconstruction.
[Kaye, Chen and Parcollet (2022)](https://doi.org/10.1103/PhysRevB.105.235115).
For a matrix causal decoder, residues must be positive semidefinite Hermitian
matrices, for example `R = L L^dagger`; elementwise positivity is insufficient.
The static term must be Hermitian. Support truncation and too few poles can
bias low-energy peaks while preserving causality. Pole locations in a compact
fit should not automatically be interpreted as physical excitations.

Nevanlinna continuation constrains an admissible family of analytic functions;
it does not make the noisy inverse problem unique.
[Fei, Yeh and Gull](https://arxiv.org/abs/2010.04572).
The TRIQS solver distinguishes diagonal NEVANLINNA and matrix CARATHEODORY
kernels and exposes Pick-matrix diagnostics.
[Solver reference](https://triqs.github.io/Nevanlinna/latest/_ref/triqs_Nevanlinna.solver.Solver.html).
Treat the static self-energy separately and supply the dynamic function in
the convention and normalization required by the selected solver. More
arithmetic precision does not replace error bars or cure incompatible noisy
data. An imaginary-axis fit error alone is not a continuation uncertainty.

A minimal future denoising experiment should use fresh ED states with known
real-axis references: compare the same continuation method on clean, perturbed
and compressed Matsubara data, with identical meshes, budgets and broadenings.
Include two nearby peaks below the available resolution and a pole near the
representation cutoff as counterexamples. Compare integrated weight, spectral
error and peak displacement across prespecified eta values. Synthetic noise
tests establish robustness to the injected noise model, not QMC robustness;
use repeated runs or a covariance estimate before making the latter claim.

### Interpretable probes do not establish disentanglement

Reuse existing bottleneck extraction before adding an encoder. Compare
train-fitted linear and modest nonlinear probes on input-side learned latents,
PCA, raw G0 features and a shuffled-label control under identical physical
group splits. Target-Sigma PCA is a descriptive compression control, not an
input-only predictor. Score probes on independent states, not six spin/orbital
blocks or three initialization seeds as independent samples.

The frozen AIM study holds U and beta fixed; it cannot demonstrate encoding of
their variation. An occupation probe is meaningful there. A low-frequency
Matsubara slope is only a proxy for Z when a Fermi-liquid regime and extrapolation
are justified; finite-bath ED does not supply a generic material quasiparticle
weight. Likewise, `-Im Sigma(iw0)` is not a measured real-axis zero-frequency
scattering rate. Iteration and warmup cannot be used as physical coordinates.

Latent rotations can change coordinate interpretations without changing the
decoder. Predictability, visual clusters and orthogonal coordinates therefore
do not establish independent physical factors. Controlled interventions or
repeated calculations at matched physical parameters are needed to separate
physics from solver artifacts. This follows the general identifiability
warning in [Locatello et al. (2019)](https://proceedings.mlr.press/v97/locatello19a.html).
With only a few material families, position the work as a controlled feasibility
and error-propagation study, not a universal materials manifold or foundation
model. The local catalog is not simply two independent oxide samples; its
candidate groups and repeated runs must be described separately.

## Basis, energy and double-counting alignment protocol

Prefer the original compatible DFT archive when its meaning can be established.
A new QE/Wannier90 or Elk Hamiltonian is not a drop-in replacement for an
unknown original correlated subspace. The following is a protocol to establish
compatibility, not a certification currently implemented by the new kernel.

1. **Define the physical calculation.** Match structure, lattice constants,
   XC functional, pseudopotential/all-electron treatment, spin/SOC convention,
   correlated atoms and orbitals, interaction U/J and temperature. Record whether
   the calculation is charge self-consistent and which charge density produced H.
2. **Establish the subspace.** Record projector/Wannier construction, local axes,
   orbital order, outer/frozen windows and normalization. Compare noninteracting
   bands, projected G0 and local crystal-field matrices in a common energy window.
   Matching eigenvalues alone does not establish matching eigenvectors.
3. **Demonstrate a basis map.** Use orbital/projector overlaps where available.
   Rotate H, projectors, Sigma, DC and any interaction tensor consistently. Test
   unitary covariance of the resulting observables. If windows/subspaces differ,
   a single local unitary may not exist; treat that case as a new model requiring
   its own calibration, not a reproduction of the public calculation.
4. **Align energy and DC.** Match units and energy origin for H, mu and Sigma.
   Record whether Sigma is before or after DC subtraction. Subtract the DC
   potential matrix once during embedding; `dc_energ` is an energy functional
   contribution and cannot replace that potential. Do not tune DC against the
   test spectrum. Compare fixed-mu errors separately from fixed-filling errors.
5. **Verify the lattice reconstruction.** Check the noninteracting limit, matrix
   causality, high-frequency normalization, finite-window weight and mesh/eta
   sensitivity; compare local G and filling to the original reference when
   available. A trace-positive spectrum alone does not prove matrix causality.
6. **Separate BZ and path products.** Use a converged weighted BZ mesh for DOS;
   a high-symmetry path gives A(k,w), not a DOS quadrature. Label the retained band
   window and spin multiplicity. A(k,w) has broadened spectral features and is
   not simply a new set of Kohn-Sham band eigenvalues.

These controls are practical: current solid_dmft documentation explicitly warns
about correlated-orbital embedding, crystal-field mismatch and using a
disentangled Wannier model with the chemical potential of a larger DMFT window.
[solid_dmft changelog](https://triqs.github.io/solid_dmft/4.0.x/ChangeLog.html).

For an aligned converted archive, use `SumkDFTTools` DOS/band postprocessing with
properly supplied real-frequency Sigma, mu and DC.
[DFTTools analysis](https://triqs.github.io/dft_tools/latest/guide/analysis.html).
For a new open-source calculation, budget an SCF/NSCF calculation, convergence
checks and Wannier construction before the dense mesh and k path. Elk task 820
also depends on a prepared ground state and Wannier-projector/path definitions;
it does not recover the original gauge from a Sigma file.
[QE interface](https://triqs.github.io/solid_dmft/4.0.x/input_output/DMFT_input/dft.html),
[Elk converter](https://triqs.github.io/dft_tools/3.1.x/guide/conv_elk.html).
Pin a mutually compatible TRIQS/converter/solver stack when executing that
phase; do not combine examples from different versions blindly.

## Incremental architecture and priorities

| Priority / decision | Reuse | Increment and boundary |
| --- | --- | --- |
| Now: can the lattice resolvent and projection be evaluated correctly? | `provenance.py`, NumPy, existing config/report layout | Implemented `lattice.py`, explicit NPZ/YAML contract, analytic tests, synthetic demonstration. No HDF5 guesses or new models. |
| Track A evidence: can a selected archive support a material reconstruction? | `hdf.py`, `triqs_inspection.py`, `validation_protocol.py`, `physics_review.example.yaml` | Fill only unresolved quantity/basis/DC/lineage evidence for a selected run; request generator metadata if needed. No repeated whole-data audit. |
| Fresh Track B generalization: does the conditional Sigma+G benefit survive a genuinely new parameter block? | `ed_workflow.py`, `ed_external_test.py`, `learning/dyson_objective.py`, `ed_report.py` | Prespecify new physical states and one holdout beyond the training support; keep validation/final test separate. Vary U/beta only in a separately declared study. Existing reserved scales lie within training range. |
| Representation interpretation | `ed_baselines._network_diagnostics`, existing trained scalers and checkpoints | Add a reusable latent extractor/probe evaluator only for a dataset with the required variable variation; no new encoder by default. |
| Continuation robustness | `PoleBaseline.predict_z`, `ed_spectra.py`, exact ED reference | Direct pole evaluation is the first reference. Add an optional external continuation adapter only after a matched ED noise/eta protocol is frozen. |
| Real-material DOS/A(k,w) | Existing selected-run audit plus verified converter | Add one versioned adapter to the normalized contract or call DFTTools directly; compare one aligned reference before expanding materials. No auto-conversion based on array dimensions alone. |

`src/` owns numerical operations; `scripts/` only composes an example;
`configs/` declares conventions; `tests/` checks independently known limits;
`reports/local/` holds generated arrays, figures and provenance without being
committed. The current addition needs no TRIQS, DFT or continuation installation.
See [the executable reference workflow](lattice_reference.md).

## Alternatives and their appropriate role

| Approach | Benefit | Limitation / best next use |
| --- | --- | --- |
| PCA/Ridge and current grid+G | Cheap, already compared; separates Sigma accuracy from property utility | No structural real-axis causal guarantee; retain as baselines, not direct DOS providers |
| Fixed real poles with positive residues | Direct complex-z evaluation; cheap scalar causal model | Pole support/resolution bias; current causal candidate underperformed; use as controlled spectral reference |
| DLR/IR compression | Compact temperature-aware imaginary-axis basis | Compression is not positivity or unique continuation; consider only if representation cost/conditioning is the question |
| Joint Sigma+G plus separately scaled tail/moment penalties | Targets resolvent sensitivity and known asymptotics | Requires verified bath and reliable moments; select weights on validation, not test; do not confuse loss terms with independent evidence |
| Nevanlinna/Caratheodory (or Padé control) | Connects admissible Matsubara data to real-frequency families | Noise sensitivity and nonuniqueness remain; benchmark on ED before a material claim |
| Direct broadened ED spectrum comparison | Known reference, avoids adding another inverse problem to the first test | Finite-bath impurity scope only; compare all states and eta dependence rather than one attractive curve |
| Original compatible DFT archive | Best opportunity for faithful lattice reproduction without a new DFT run | Depends on projector/DC provenance and a compatible runtime |
| New QE/Wannier90 or Elk model | No commercial DFT fee; independent, reproducible Hamiltonian | Additional convergence/subspace work; without demonstrated alignment it is a new exploratory model |

The research contribution to test is whether a compact representation preserves
observables under specified shifts with a favorable accuracy/cost tradeoff.
It is not the already established G0-to-Sigma mapping, causality by itself, or
an unvalidated plot of a material DOS.

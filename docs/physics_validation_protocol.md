# Physics validation protocol and figure conventions

This document specifies the next research checks; it does not report them as
executed on the real dataset. The user's local exploratory run reported 62 runs
in 9 candidate groups, with final BVO arrays equal to stored iteration 5 and
large later stored-array changes. Those observations motivate reference review.
The selected 64-frequency window, unreviewed energy/basis conventions and unknown
solver uncertainty currently limit physical interpretation.

## Order of work and evidence to retain

| Order | Question | Work and retained evidence | Interpretation limit |
|---|---|---|---|
| 1 | What are the inputs and targets? | Match code/input/publication definitions to each raw path; record DFT projected G0 versus Weiss, basis, U/J, filling, mu and DC conventions. | Names or scalar metadata alone do not establish physical meaning. |
| 2 | What happened near the large stored change? | Compare all numbered arrays, signed changes per block, and original log/code stages around iterations 5–9; record restart/mixing/mu/DC provenance. | Adjacent numeric labels may not mean identical pre/post-solver stages. |
| 3 | How uncertain is the reference? | Request independent seeds and solver bin/autocorrelation information at fixed reviewed physical conditions. Check up/down and orbital differences only against justified symmetries. | Warmup/iteration variations are correlated records, not independent error bars. |
| 4 | Are same-stage impurity quantities consistent? | Use reviewed Weiss, G_imp and uncorrected solver Sigma; save matrix Dyson residuals and inversion-conditioning sensitivity. | Small algebraic residuals do not establish lattice self-consistency. |
| 5 | Are static and dynamic terms reliable? | Compare stored density/interaction moments with wider-frequency tails. Declare several fitting windows/orders before fitting and retain residuals and variation. | Do not call the first-window mean Sigma-infinity or silently shift target energies. |
| 6 | Did DMFT reach a physical fixed point? | With H0/projectors available, compare G_imp and G_loc in the same basis, filling, mu and relevant energies across complete cycles; record noise and mixing alongside residuals. | Fixed iteration counts, final equality and one declining norm do not certify convergence. |
| 7 | Does prediction preserve the intended observables? | Freeze validation criteria, then compare reference versus predicted filling, orbital polarization and eligible metallic Z estimates. Independently validate continuation before lattice spectra/DOS. | A global Matsubara error or necessary sign check does not certify spectral accuracy. |

The diagnostic command writes `validation_plan.json` and a compact status table
in `summary.md`. `observed` means a descriptive calculation was performed;
`researcher_attested` refers to supplied evidence; neither means independently
verified physics. The plan does not block exploratory execution.

## Energy and frequency checks

Confirm native units before converting Sigma or omega to meV. Define the error
budget for an observable, not by copying DFT SCF total-energy tolerances. For a
proposed sensitivity study, perturb static and frequency-dependent Sigma terms
separately at several declared scales (for example 1, 5 and 10 meV after unit
review), recompute the chosen observable using identical H0/DC conventions,
and compare to reference uncertainty. These are exploratory perturbation scales,
not community acceptance thresholds. Preserve fixed filling or explicitly record
when mu is held fixed; these are different experiments.

For beta=40 in reciprocal-eV units, the first positive Matsubara point is about
0.0785 eV and the 64th is about 9.975 eV. The global norm mixes low and high
energy behavior. Compare the same physical windows across temperature studies,
record endpoints, avoid interpolation unless independently tested, and do not
interpret a finite-temperature first-point estimate as a zero-temperature
derivative. Orbital shifts relative to other bands are not automatically removed
by a chemical-potential adjustment.

For moment tests, request stored Sigma_Hartree/Sigma_moments and the interaction
and density-matrix conventions used to generate them. A tail fit without moments
is an estimate requiring window/order stability checks, not an independent
certificate. None of these quantities is fabricated when absent.

## Figure specification

Plots use local Matplotlib settings (no global style mutation), physical omega
axes, explicit units, consistent fonts, panel labels and separate draft captions.
Preview PNGs use 300 dpi; companion PDFs retain vector lines/text. An accessible
palette is paired with dashed lines or marker shapes where appropriate. Confirm
grayscale readability and labels at intended column width before submission.
These are journal-style starting points, not a guarantee of a journal's current
submission compliance.

- Prediction overview: original Re/Im curves and signed errors, with no oracle
  centering. A block-resolved companion separates blocks that overlap in the
  overview; metrics still include all matrix elements.
- Iteration trajectories: one block per row, Re/Im columns, numeric iteration
  colorbar and black dashed final array. No spin/orbital averaging. For matrix
  blocks all diagonal elements are displayed; off-diagonal numerical errors are
  not represented by these diagonal plots.
- Adjacent changes: separate constant-real, varying-real and imaginary curves;
  block-wise complex changes alongside them. Linear axes preserve zero changes.
  Numbers refer to previous **stored** iterations; missing iterations are never
  filled. A future log view must disclose handling of zeros explicitly.

Draft `.caption.md` files describe the panel meaning, window/units where
available, and descriptive scope. Use them with the source config, data hashes,
block CSV and diagnostic JSON. Do not add standard-error shading based on the
three deterministic nearest-neighbor seeds or correlated channel replicas.
Do not tune on the previously inspected BVO test group and present that group
again as untouched final validation.

## Source references

- [APS guidance on color and grayscale figures](https://journals.aps.org/authors/guide-acceptable-color-online-figures-h24): distinguish curves beyond color alone.
- [APS author style guide](https://res.cloudinary.com/apsphysics/image/upload/APS_Journals_Style_Guide_Authors022326_guzwhx.pdf): panel labels, captions and accessible figure design.
- [solid_dmft output observables](https://triqs.github.io/solid_dmft/3.3.x/input_output/DMFT_output/results.html): occupancy, mu, energy and iteration-change records.
- [CTHYB moments](https://triqs.github.io/cthyb/latest/guide/high_freq_moments.html): density/interactions and stored self-energy moments.
- [VASP EDIFF](https://vasp.at/wiki/index.php/EDIFF): SCF energy stopping criterion, distinct from self-energy approximation error.

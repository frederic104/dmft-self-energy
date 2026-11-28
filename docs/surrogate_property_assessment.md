# Surrogate assessment and reconstructed properties

Reviewed 2026-11-28 after PR #48. This is an interpretation and saved-result
reconstruction of the existing study, not a new model-selection experiment.
Candidate search remains closed. The [closeout](track_b_closeout.md) contains
the full scores and configuration; this note connects them to research decisions.

## What is being evaluated?

The question is whether predicting a compact representation of self-energy
preserves useful observables better than a simple baseline. PCA8 is the main
Sigma baseline; the grid Sigma+G model tests a property-oriented alternative;
the uniform-pole causal model tests a constrained parameterization. The physical
unit in Track B is a complete AIM parameter tuple, not a frequency, spin channel,
or initialization seed. The next decision is which existing result is defensible
and whether a material-property extension is supported, without retuning.

Track A convergence was already unresolved when that track was established.
It is a standing dataset limitation, not a newly discovered reason to restart
auditing. Agreement with an archived target can be measured without asserting
that the target is a converged physical solution. The existing Track A scores
answer the former question. Controlled ED supplies the independent physical
reference for the property questions in Track B.

Self-energy prediction from G0 is established prior work; see
[Mitra and Banerjee](https://doi.org/10.1038/s41524-026-02277-6) and the
[source review](upstream_data_contract.md). This study does not establish a new
transfer-learning method. Its present result is a comparison of representation,
prediction error and downstream property preservation in a small AIM family.

## What each completed stage means

| Stage | What it establishes | What it does not establish |
| --- | --- | --- |
| Track A inventory/extraction | Accessible complex arrays, meshes, paths and provenance | Meaning of G0 or convergence of the archived target |
| Track A grouped baselines | Exploratory G0-1 to Sigma_iw mapping across candidate material/beta groups | Reproduction of the upstream transfer-learning protocol or verified material generalization |
| Finite-bath ED and limiting-case checks | Controlled impurity G, Sigma, thermal occupation and Lehmann spectrum | Full lattice DMFT or a material DOS |
| Fixed-bath random/blocked epsilon | Interpolation sensitivity to the chosen holdout | Broad physical generalization; epsilon_linear also has different inputs |
| Bath variation and PCA/Ridge | Compressibility and a strong simple prediction baseline | An intrinsic physical latent dimension of eight |
| Matched checkpoint refinement | Conditional benefit of adding a G objective at matched extra budget | A universal effect of G loss across models and axes |
| Decoder/pole diagnostics | Distinction between target approximation and learning coefficients from G0 | An isolated test of causality; parameter counts differ |
| Frozen epsilon/reserved scale | Performance of frozen choices on the specified held-out states | New U, beta, bath size, joint-axis or material extrapolation |
| Dyson and occupation reconstruction | How Sigma error propagates into impurity G and n | A learned standalone property predictor or analytic continuation |
| Causal impurity spectra | Real-z evaluation against identically broadened ED | Spectral accuracy of PCA/grid, or lattice DOS accuracy |

The reserved scale 1.15 lies inside the training range. Each axis has 40 reserved
physical states; seeds 31/32/33 are repeated optimizer initializations on those
same states. The two axes are separate experiments. All 12 final neural fits
stopped at their 100-iteration budget, not demonstrated optimizer convergence.
That optimization issue is separate from the unresolved Track A DMFT convergence.

## Quantitative surrogate decision

Reserved scale, mean of seed-level RMSEs. G uses the first eight positive
Matsubara frequencies; n is per spin and compared with exact thermal ED.
Native model units are used, with no conversion to material eV.

| Axis | Model | Sigma RMSE | G RMSE | n RMSE |
| --- | --- | ---: | ---: | ---: |
| coupling | PCA8 | 0.021437 | 0.082703 | 0.014427 |
| coupling | grid Sigma+G | 0.033492 | 0.086566 | 0.014187 |
| coupling | causal Sigma | 0.051119 | 0.218738 | 0.049010 |
| energy | PCA8 | 0.025639 | 0.101919 | 0.018359 |
| energy | grid Sigma+G | 0.046874 | 0.070507 | 0.014362 |
| energy | causal Sigma | 0.088100 | 0.284098 | 0.064855 |

Relative to PCA8, grid Sigma+G changes mean Sigma/G/n RMSE by
**+56.24% / +4.67% / -1.66%** for coupling and
**+82.82% / -30.82% / -21.77%** for energy. These are ratios of mean RMSEs,
not means of statewise percentage changes. Seed variability matters: energy
occupation RMSE at seed 32 is 0.022308, worse than PCA's 0.018359. The full SDs
and unseen-epsilon results remain in the closeout; the latter are not pooled
with this table.

- **Sigma surrogate:** retain PCA8. The learned/constrained alternatives do not
  improve this baseline in the frozen comparisons.
- **Impurity property surrogate:** grid Sigma+G is conditionally useful for
  energy-axis G; the n improvement is less stable. A smaller Sigma RMSE alone
  does not guarantee a smaller property error.
- **Causal representation:** retain it as a control. Nonnegative spectra and
  near-correct integrated weight coexist with sizable spectral shape errors.
  There is no evaluated PCA/grid real-frequency comparator.
- **Computational replacement:** not established. These reports do not provide
  a matched end-to-end timing benchmark, training-cost amortization or a
  prespecified application error tolerance. Do not claim a measured speedup,
  solver replacement, or deployment readiness from accuracy tables alone.

## Reconstructed values, not only scores

The updated `ed_report.frozen_diagnostics` exports reference/predicted G and
Sigma at omega_0, raw/tail-corrected occupations, and signed occupation-error
components in `state_errors.csv`. No model is loaded; G and n are recomputed
from saved Sigma with the existing Dyson and tail routines and checked against
the saved G/n arrays. The original ED reference arrays are reused, not regenerated.

The following display states are selected by epsilon order only: smallest,
upper middle (index 20 of 40 sorted values), and largest. All 40 states per axis
remain in the CSV. PCA is deterministic; ranges below cover all three model
seeds and are not confidence intervals. Every value is a per-spin occupation.

| Axis | epsilon_d | Exact ED n | PCA8 n | Grid n: min to max | Causal n: min to max |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupling | -0.976817 | 0.495257 | 0.473257 | 0.481566 to 0.523460 | 0.521014 to 0.597365 |
| coupling | -0.282458 | 0.416753 | 0.419308 | 0.421101 to 0.433038 | 0.413641 to 0.423864 |
| coupling | 0.193291 | 0.266191 | 0.262522 | 0.250987 to 0.270453 | 0.201114 to 0.221599 |
| energy | -0.976817 | 0.484639 | 0.490220 | 0.484398 to 0.513546 | 0.481585 to 0.514959 |
| energy | -0.282458 | 0.435364 | 0.427044 | 0.416493 to 0.438724 | 0.368265 to 0.375645 |
| energy | 0.193291 | 0.209218 | 0.248220 | 0.194837 to 0.231055 | 0.157500 to 0.171349 |

For a worked example, use the energy-axis upper-middle state, PCA8, seed 31:
epsilon_d=-0.2824583953, beta=20, omega_0=pi/20=0.1570796327.

| Quantity | ED reference | PCA reconstruction |
| --- | ---: | ---: |
| Sigma(i omega_0) | 0.0179380203 - 0.6416509561i | 0.0621084989 - 0.7402734213i |
| G(i omega_0) | 0.4677190546 - 0.8818566994i | 0.3702887579 - 0.8565608896i |
| Finite 64-frequency n sum | 0.4446288647 | 0.4362732667 |
| Tail-corrected n | 0.4353669726 | 0.4270443875 |
| Exact thermal n | 0.4353638465 | Not used in prediction |

With deltaG=G_pred-G_ref, define L=(2/beta) sum_first8 Re(deltaG),
H=(2/beta) sum_remaining Re(deltaG), T as the predicted minus reference
tail correction, and B=n_ref_tail-n_ED. Then, state by state,

```text
n_pred_tail - n_ED = L + H + T + B
```

For this example the terms are -0.008550312972, +0.000194714926,
+0.000033012925, and +0.000003126121, summing to -0.008319459001.
The low-frequency error dominates but the other terms partially cancel it.
Signed terms add; their RMSEs do not. The reference tail-versus-exact RMSE is
4.071e-6 for coupling and 3.113e-6 for energy, substantially below the model
occupation errors here. This is a check of the reference estimator, not a
guarantee for arbitrary learned high-frequency behavior.

## Track A: retain the existing interpretation

The existing Ubuntu `exploratory_v1` and `track_a_20261126_120723` metrics both
record 62 run files aggregated into nine candidate material/beta groups:
seven training groups, CVO/beta40 validation and BVO/beta40 test. Their input is
`/dmft_output/G0-1`; their target is `/dmft_output/Sigma_iw`. Both select nearest
neighbor, with validation complex RMSE 0.05294503 and BVO complex RMSE 0.07686763
(imaginary RMSE 0.01348081). These are existing reported metrics, not newly
recomputed predictions. Seeds 0/1/2 give identical nearest-neighbor results.

Both reports record the same prepared-array SHA-256:
`705d06216f6999a63c11a0ad13230ff69c2ddd00855eb372a3091a07c1614080`.
They also have the same split and scores. They are not independent evidence
from two datasets. Nine provisional groups are not nine verified physical-state
identities; warmup variants do not supply independent physical samples.
Material labels, units, common basis, G0 meaning and target convergence remain
unresolved. Do not report these errors in meV or reinterpret them as errors
against verified converged self-energy.

Metrics-file SHA-256 values, recorded during this read-only review:

```text
exploratory_v1/metrics.json
653ca015bc9d2030f119f37a15ae42665c2da1c9e1db82606f50124f690bac76
track_a_20261126_120723/metrics.json
20397f02040155e866717dab28d3e500e7f9e6166aaf5f8a95d7f012bfb5cb43
```

### One nominated material record: payload exists, interpretation is incomplete

Only `SVO/svo_beta40_nwarmup100000.h5` was opened for this follow-up. Its SHA-256
is `4cfff84764d8f13b9fafd8f4aeef6fcd771c87e8f4eb43e347befa050b2e6526`.
SVO is the archive's candidate label. No raw or processed HDF5 was copied,
modified or uploaded. Stored observations are:

| HDF5 path | Observed value or stored shape | Interpretation boundary |
| --- | --- | --- |
| /dmft_output/G0-1..25, G-1..25, Sigma-1..25 | Numbered groups 1 through 25 for each quantity | Labels do not prove solver/mixing stage |
| /dmft_output/iterations | 55 | Does not match the numbered-group maximum; no restart/history inference |
| /dmft_output/G_iw, G_tau, Sigma_iw | Final-named groups present | Final naming does not certify convergence |
| /user_data/chemical_potential | 0.024190458378013074 | Energy convention and stage unverified |
| /user_data/dc_imp/0/up and down | 3x3 diagonal, entries 1.3514590512895583 | Application to stored Sigma and stage unverified |
| /user_data/dc_energ/0 | 0.0007299198726801131 | Stored scalar, not a substitute for the DC matrix |
| /dft_input/energy_unit | 13.605698 | Stored schema value, not proof of units of every archived array |
| /dft_input/hopping | [10,1,5,5,2] | Payload present; encoding and conventions need reviewed use |
| /dft_input/proj_mat | [10,1,1,3,5,2] | Payload present; basis/embedding alignment unverified |
| /dft_input/bz_weights | Ten values, sum 1 | Normalization does not establish k-grid convergence |
| /dft_input/SP, SO, use_rotations | 0, 0, 1 | Stored flags; rotations must not be silently ignored |

Thus lattice ingredients are not simply absent from the nominated file.
Their presence does not resolve the missing contract. In particular, the
existing name-based metadata scanner reported only `iterations` and
`chemical_potential`; DC values reside below groups and were found by explicitly
reading those nominated paths. Absence from that scanner's output is not proof
that a physical field is absent from HDF5.

The local physics review still marks definitions unresolved and contains no
reviewed Dyson specification. Calling the existing `physics.dyson_check` with
an unresolved specification returned `not_assessed`: no numerical physical
Dyson validation was executed. This is not a failed Dyson identity or evidence
of nonconvergence. No verified material G, occupation, DOS or Z is reported.

This completes the bounded feasibility check. Do not repeat full-data audits to
resolve missing producer information. If authoritative assignments and run-level
evidence later become available, update the existing per-run review, establish
basis/units/mu/DC/stage, and reconstruct **reference** G/n before evaluating a
material surrogate. Lattice reconstruction requires k-resolved inversion and
projection, rather than replacing the Weiss function by an averaged DFT G0;
see [DFTTools' formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html).
Without that evidence, continue to describe Track A as exploratory stored-array
prediction and use Track B for the controlled property conclusion.

## Reproduction and executed checks

Use the existing [report-only command](closeout_reproduction.md), with a new
output directory `reports/local/surrogate_property_assessment_v1`. Existing
result directories are the inputs; no training command is needed. This follow-up
changes only the reporting fields and interpretation, not trained models,
splits, selection rules or physical formulas.

Executed against the existing saved results: 34 input artifacts, zero missing
inputs, 18 output files, including eight existing plot types. All saved frozen
scores agreed with recomputation. The CSV contains 960 state/model/seed rows:
240 from unseen epsilon and 720 from reserved scale. Those row counts do not
represent independent physical samples. Maximum absolute residual of the signed
occupation decomposition over all rows was 1.53e-16.

Six relevant tests passed on Windows Python 3.12.7: `tests/test_ed_report.py`
and the two analytic density/Dyson tests listed in the reproduction guide.
The reporting test independently checks a shifted noninteracting level against
the Fermi function and analytic complex G, while forbidding model loading,
fitting and prediction. The real-result run checks the available frozen arrays;
neither test recreates historical training or verifies material convergence.

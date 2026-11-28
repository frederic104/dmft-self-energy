# Track B closeout: representations, Dyson errors and impurity properties

Reviewed 2026-11-28 against main commit a73902e (frozen impurity spectra).
Candidate search remains **closed**.

The evidence supports **PCA8 as the Matsubara self-energy baseline**, grid
Sigma+G as a conditional property-oriented alternative, and the present uniform
causal model as a control rather than the principal predictor. This is a decision
about this limited AIM family, not a rejection of causal representations generally.
Close this study without further model, loss-weight or pole searches.

## Evidence and verification scope

The handoff numbers were initially user-reported. Existing Ubuntu JSON/CSV
reports and frozen NPZ predictions subsequently became accessible through WSL.
The frozen results below were **recomputed from saved predictions and checked
against the original summaries and handoff**. Development results were read from
existing summaries/metrics, without recomputing their predictions. No model was
retrained, no checkpoint inference was rerun, and full ED datasets were not
regenerated. This is saved-result verification, not independent reproduction of
training or data generation.

The reporter reuses compare_properties, occupancy_estimates and spectrum_metrics.
It checks frozen dataset hashes, reference/parameter ordering, recomputed scores,
and spectral/Matsubara report links. Newly recorded prediction hashes identify
files read now; they are not original creation-time signatures. Model bundles
are not loaded.

| Stage | Existing directory under reports/local/ | Question and scope |
| --- | --- | --- |
| Fixed-bath development | linear_frequency_random_seed31; linear_frequency_blocked_seed31 | Random versus blocked epsilon; already used for development |
| Bath development | aim_properties_20261127_144210 | Validation scale .85; 1.15 then reserved |
| Matched checkpoint refinement | green_objective_20261127_191941 | Additional optimization versus adding G loss |
| Decoder x loss | decoder_comparison_20261127_200055 | Reset grid/causal heads, Sigma/Sigma+G, seed 31 |
| Pole diagnostic | pole_resolution_20261127_203835; causal_low_20261127_203835 | Training direct fit versus validation prediction |
| Frozen unseen epsilon | frozen_epsilon_v1 | 40 new epsilon values/axis, data seed 441, previously used scale .85, model seed 31 |
| Reserved | final_reserved_v1 | Scale 1.15, original matched epsilon values, model seeds 31/32/33 |
| Spectra | final_spectra_v1 | Same reserved states/checkpoints; not independent samples from the Matsubara evaluation |

## Configuration and physical unit

Spinful single impurity plus two bath sites, 64-dimensional Fock space, U=2,
beta=20, mu=0, 64 positive Matsubara frequencies. Dataset seed 31 draws 40
epsilon_d values from [-1,.2]. Base bath energies are (-1,1), couplings (.5,.7).
Coupling scales V only; energy scales bath energies only. These are separate
experiments, not a joint grid. Scales=[.7,.85,1,1.15,1.3], train=[.7,1,1.3],
validation=.85, reserved=1.15: 120/40/40 states per axis, with matched epsilon
values across scales.

The physical unit is the full AIM parameter tuple. Three seeds are optimizer
initializations on the same 40 states, not 120 independent test states. Scale
1.15 is parameter interpolation, not range extrapolation. New-epsilon evaluation
also stays within the same U/beta/bath family. Spin-averaged quantities are per
spin; neither spin replicas nor model seeds increase the physical sample count.

All frozen models receive 128 real/imaginary G0 features. PCA8 is target-side
PCA plus Ridge(alpha=1), not a learned input latent encoder. Networks have ReLU
hidden widths (64,8,64). Grid/causal parameter counts are 17,672/10,457. The
causal head predicts a real constant and 16 positive residues on uniform real
poles in [-4,4]. Latent dimension 8 and pole count 16 differ. Decoder comparisons
therefore change capacity/parameterization, not causality alone.

All 12 final neural fits (2 axes x 3 seeds x 2 models) have iterations=100,
success=False, STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT in the saved records.
These are fixed-budget solutions, not demonstrated converged optima. PCA is
reused deterministically. SD uses numpy.std with ddof=0 across model seeds;
a single seed has unavailable SD. PCA SD=0 does not mean zero generalization
uncertainty. Means average **seed-level RMSEs**, not ensemble predictions.

## Development evidence

Fixed-bath epsilon_linear validation Sigma RMSE was .001086 (random) and
.073345 (blocked). It receives epsilon_d directly, while other models receive
G0. This is not an equal-input comparison or evidence of broad latent structure.

Bath-validation Sigma RMSE for PCA8/full-target Ridge was .024126/.024133
(coupling) and .019173/.019168 (energy); PCA16 was effectively identical to
Ridge. This supports compressibility here, not intrinsic physical dimension 8.

Matched refinement from the same checkpoint with the same additional budget:

| Axis/model | First-eight G RMSE: Sigma-only to Sigma+G | n RMSE: Sigma-only to Sigma+G |
| --- | --- | --- |
| coupling direct MLP | .252753 to .109795 | .056810 to .023857 |
| coupling latent8 | .053771 to .055517 | .008070 to .009017 |
| energy direct MLP | .077028 to .034232 | .019917 to .006270 |
| energy latent8 | .076138 to .050834 | .018543 to .005163 |

The G term helped conditionally. Final PCA-versus-grid scores alone cannot
identify its causal effect. In the reset decoder comparison, energy causal G
RMSE worsened from .096316 to .109905 when adding G loss.

Low-frequency poles reduced training direct-fit RMSE from .058300 to .002613
(coupling) and .052506 to .002403 (energy). Causal Sigma-only validation
changed from .074959 to .071029 and .031269 to .037224; occupation worsened
from .025457 to .035809 and .009030 to .017798. Direct fitting accesses each
training target, whereas prediction learns coefficients from G0 and transfers
to validation. Their gap cannot all be attributed to optimization failure.
Direct fitting minimizes standardized-coordinate least squares, so its physical
RMSE is not a mathematical lower bound in that metric. Physical admissibility,
reference approximation and coefficient learnability are distinct. The final
causal model keeps the original uniform poles.

## Frozen evaluation

New epsilon, model seed 31 only (no seed SD):

| Axis | Model | Sigma RMSE | First-eight G RMSE | Per-spin n RMSE |
| --- | --- | ---: | ---: | ---: |
| coupling | PCA8 | .026627 | .147514 | .032766 |
| coupling | grid Sigma+G | .049273 | .057555 | .011502 |
| coupling | causal Sigma | .086398 | .172148 | .027674 |
| energy | PCA8 | .019346 | .110217 | .021180 |
| energy | grid Sigma+G | .032708 | .067792 | .014222 |
| energy | causal Sigma | .033804 | .099133 | .010460 |

Reserved scale 1.15, mean +/- population SD across model seeds:

| Axis | Model | Sigma RMSE | First-eight G RMSE | Per-spin n RMSE |
| --- | --- | ---: | ---: | ---: |
| coupling | PCA8 | .021437 +/- .000000 | .082703 +/- .000000 | .014427 +/- .000000 |
| coupling | grid Sigma+G | .033492 +/- .003713 | .086566 +/- .010679 | .014187 +/- .002105 |
| coupling | causal Sigma | .051119 +/- .009817 | .218738 +/- .026958 | .049010 +/- .009990 |
| energy | PCA8 | .025639 +/- .000000 | .101919 +/- .000000 | .018359 +/- .000000 |
| energy | grid Sigma+G | .046874 +/- .006136 | .070507 +/- .012395 | .014362 +/- .005787 |
| energy | causal Sigma | .088100 +/- .002865 | .284098 +/- .022290 | .064855 +/- .007627 |

PCA retains its Sigma advantage. Grid improves energy G at every seed, but
seed 32 occupation RMSE is .022308 versus PCA .018359. The tiny mean coupling
occupation change is not consistent improvement. Retuning from these scores
would turn subsequent work into development; these states could not then be
reused as an independent final test.

## Error propagation checked on all saved states

For fixed G0, deltaSigma=Sigma_pred-Sigma_ref, the scalar identities are:

~~~text
G = 1/(1/G0 - Sigma)
deltaG = G_pred G_ref deltaSigma
       = G_ref^2 deltaSigma / (1 - G_ref deltaSigma)
deltaG_linear = G_ref^2 deltaSigma
deltaG - deltaG_linear = (G_ref deltaSigma) deltaG
~~~

Linearization is useful for small r=|G_ref deltaSigma|. Wherever deltaG is
nonzero, r is exactly the pointwise relative linear remainder. Large |G|
amplifies the same Sigma residual; total Sigma RMSE does not capture this
state/frequency sensitivity. Occupation uses a signed Re G sum and tail, so
G RMSE also does not uniquely determine occupation error.

All 40 reserved states, ranges across seeds:

| Axis/model | Relative L2 linear remainder | Maximum r |
| --- | ---: | ---: |
| coupling PCA8 | .263 | .373 |
| coupling grid Sigma+G | .227-.497 | .330-.745 |
| coupling causal Sigma | .314-.536 | .369-.835 |
| energy PCA8 | .236 | .293 |
| energy grid Sigma+G | .227-.308 | .300-.469 |
| energy causal Sigma | .501-.580 | .595-.674 |

Relative L2 is ||deltaG-deltaG_linear||_2 / ||deltaG||_2. Exact identity residuals
were below 6.3e-16 across reserved cases. First-eight frequencies carried at
least 99.93% of G squared error in every case, versus 78.8-99.3% for Sigma.
A low-frequency explanation is supported, but the linear remainder is too
large to replace exact propagation. Percentiles over all frequencies can hide
the few important low-frequency failures.

State plots/CSV use all epsilon_d and exact ED occupations. Energy PCA
occupation absolute-error median/90th percentile/maximum:
.007932/.032518/.057610. Grid values: .005804/.011116/.027946 (seed 31),
.019300/.032123/.049106 (32), .003373/.025867/.030904 (33). Aggregate improvement
does not imply statewise dominance. Grid beats PCA on first-eight G for
67.5/17.5/32.5% of energy states and 20/35/17.5% of coupling states across
seeds 31/32/33. Reducing some large PCA errors matters to the energy aggregate.
These are post-hoc descriptive proportions, not significance tests or selection.

Plots average seed diagnostics separately at each physical state; CSV preserves
every seed. IDs hash the full parameter tuple. Squared-error differences pair
the same state/seed with PCA. Epsilon and filling are related coordinates,
not independent causal factors. No favorable subrange becomes representative
performance; seeds are never used as independent physical samples for intervals.

### Implemented loss

For N training states, M frequencies and training SD s_j of each real/imaginary
target coordinate j:

~~~text
L = mean_{N,2M} [(Sigma_pred,j-Sigma_ref,j)/s_j]^2
  + lambda * mean_{N,M}|G_pred-G_ref|^2 / mean_train|G_ref|^2
  + alpha/(2N) * sum_layers ||W||_F^2
~~~

Existing training-only StandardScalers are reused. Means cancel in residuals;
real and imaginary coordinates are scaled separately. Lambda=1 for the grid
finalist, 0 for causal; alpha=.001. G loss uses all 64 frequencies and a single
training norm, not only eight points or separate per-frequency scaling.
For small residuals it weights approximately |G_ref|^4|deltaSigma|^2. The actual
gradient uses G_pred^2. This is a property objective derived through Dyson,
not a new independent physical law. No direct occupation/moment/spectral loss
is imposed.

## Impurity spectra: admissibility versus shape accuracy

Window [-8,8], 1601 points, eta=.05/.1; seed means +/- SD:

| Axis | eta | A RMSE | Mean integrated absolute error | Predicted/ED window weight | n RMSE: broadened ED / exact ED / ED bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupling | .05 | .181829 +/- .001642 | .980620 +/- .004894 | .995869/.995867 | .029484/.035996/.013890 |
| coupling | .10 | .092539 +/- .001762 | .594971 +/- .005179 | .991739/.991735 | .019940/.030292/.020895 |
| energy | .05 | .194931 +/- .003950 | .988551 +/- .011669 | .995886/.995886 | .045004/.059094/.020684 |
| energy | .10 | .103185 +/- .002063 | .627100 +/- .008729 | .991772/.991772 | .032238/.057877/.033440 |

Status counts: 12 ok, 24 unsupported. PCA/grid have no validated real-frequency
representation; their spectral performance is unevaluated, not ranked.
Negative-spectrum fractions are zero. Sigma_infinity RMSE is .017200/.008475
for coupling/energy; 1/z coefficient RMSE is .042761/.007047. Moments do not
depend on eta. All-state spectral RMSE and integrated-error distributions are
exported rather than relying on the first-state example.

Integrated absolute error .98 is not “98% wrong.” Matching total weight permits
different peak positions, widths and weight allocation. Lower RMSE at larger
eta is smoothing, not model improvement. Do not linearly subtract RMSEs against
different references to isolate model error. The old README's approximate .596
coupling error at eta=.1 differs from the saved .594971 used here.

## Physical definitions and the next property stage

H contains epsilon_d, bath energies and U n_up n_down; ED diagonalizes K=H-mu*N.
Complex z and real energy are relative to mu; G0 uses z+mu. Mu=0 here, with no
double counting. No arbitrary conversion to eV is made. Under spin symmetry,
stored spin averages are per-spin values and total occupation is twice n.

~~~text
n_spin = 1/2 + (2/beta) sum_{n>=0} Re G(i omega_n)
A_imp(E) = -Im G_imp(E+i eta)/pi
n_spin = integral f_beta(E) A_imp(E) dE  (eta->0, full range)
Sigma(z) = U n_opposite + U^2 n_opposite(1-n_opposite)/z + O(z^-2)
~~~

The Matsubara implementation fits Re G=-m1/w^2+m3/w^4 on the last min(16,M/2)
points (16 here), then analytically sums omitted terms. No true occupation or
moment enters the predicted tail. Reserved ED-reference tail-versus-exact RMSE
is approximately 3-4e-6; that does not certify every learned tail. Values are
not clipped. A canonical single-orbital per-spin spectrum integrates to one
over the full line; finite eta/window occupation bias is checked against ED.
The Sigma moments above apply to this single-orbital interaction, not directly
to a multi-orbital Kanamori model.

| Property | Required inputs and reference | Present capability |
| --- | --- | --- |
| Matsubara impurity G, n | Same AIM G0 and predicted Sigma, mesh/beta; ED G/thermal n | Existing three models plus Dyson/tail; no new model needed |
| Impurity A | Real-z Sigma and finite-bath G0; identically broadened Lehmann ED | Causal only; PCA/grid continuation needs separate validation |
| Material local/orbital A | H(k), projectors/embedding, basis, mu, DC, real-z matrix Sigma; verified DMFT reference | Meanings/payload unresolved; unavailable now |
| Lattice DOS | Above plus k weights, represented bands, overlap S(k) if needed | Different from AIM impurity A; unavailable now |
| Z/scattering rate | Suitable low-temperature/low-energy limit and validated reference | Not reported as reliable material observables here |

One explicit orthogonal Bloch-basis convention:

~~~text
Sigma_emb(k,z) = P(k)^dag [Sigma_imp(z)-V_DC] P(k)
G_k(z) = [(z+mu)I-H_KS(k)-Sigma_emb(k,z)]^-1
G_loc(z) = sum_k w_k P(k) G_k(z) P(k)^dag
DOS(z) = -Im sum_k w_k Tr G_k(z) / pi
~~~

Nonorthogonal bases require S(k) and the corresponding trace convention.
**Applying [G0_local^-1-Sigma]^-1 after k-averaging DFT G0 does not generally give
lattice local G.** Matrix inversion and k averaging do not commute. Impurity
Dyson needs the appropriate Weiss function. This is a practical reason to resolve
Track A input semantics first, as distinguished in the
[DFTTools formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html).

When Fermi-liquid assumptions hold, Z=[1-d ReSigma^R/dE at zero]^-1 and one
quasiparticle-width convention is -Z ImSigma^R(0). A two-bath finite system at one
beta does not establish that limit. A first-Matsubara estimate or a derivative
at an arbitrary eta is not automatically a material quasiparticle weight/lifetime.

## Track A, novelty and bounded follow-up

See [upstream_data_contract.md](upstream_data_contract.md) for checked evidence,
unresolved meanings and the minimum contract. Track A remains exploratory;
verified DFT-G0-to-converged-Sigma and material DOS claims are unsupported.

[Mitra and Banerjee (2026)](https://www.nature.com/articles/s41524-026-02277-6)
already address DFT-derived G0-to-Sigma prediction.
[Dong, Gull and Wang (2024)](https://link.aps.org/accepted/10.1103/PhysRevB.109.075112)
predict Lehmann self-energy matrices with an equivariant network and connect
them to G/observables through Dyson and an electron-number condition. Neither
the mapping, causal representation nor Dyson reconstruction alone is novel.
[Liebsch and Ishida](https://arxiv.org/abs/1109.0158) study temperature/bath-size
accuracy in ED DMFT; that work does not certify this fixed AIM's lattice accuracy.

The supported contribution is a reproducible controlled benchmark separating
compression, objective and decoder effects, including why Sigma rankings differ
from property/spectral accuracy. “Causality does not guarantee accuracy” is not
a new theoretical discovery. Given the narrow family, limited advantage over
baselines and budget-limited fits, evidence for a standalone methods paper is
weak. A benchmark with documented negative results and a foundation for follow-up
is more defensible. Compression runtime/memory advantages also remain unmeasured.

Next: preserve this report/models/splits and close Track B; obtain Track A
generation assignments and convergence evidence for one run; then reconstruct
existing reference G/occupation for one verified material to check conventions.
The stopping condition is a linked record of input/target meaning, mesh, basis,
mu/DC, physical-state identity and convergence evidence. If evidence stays
unavailable, retain exploratory status and stop material-performance claims.
No new architecture, U/beta sweep or bath expansion is required now. Any later
experiment must specify its claim, why existing results cannot answer it and a
stopping criterion before inspecting a new final set.

## Reproduction and privacy

See [closeout_reproduction.md](closeout_reproduction.md). The reporter reads
existing results, never opens model joblib files, records missing arrays, and
fails on inconsistent provenance/scores. Output directories are not overwritten.

Shared outputs use run labels, relative file names and hashes instead of personal
absolute roots. The committed TRIQS report's 63 source locations and its producer
were also redacted. Existing experiments, raw HDF5 and synced project sources
remain unchanged. Working-tree edits do not remove paths from historical commits.


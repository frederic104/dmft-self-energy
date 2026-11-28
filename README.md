
# DMFT Self-Energy

## Current result: Track B candidate search is closed

The [integrated closeout](docs/track_b_closeout.md) separates development,
frozen unseen-epsilon, reserved-scale and impurity-spectral results. Existing
Ubuntu frozen predictions were read and their metrics recomputed; no training
or checkpoint inference was repeated. PCA8 remains the Sigma baseline, grid+G
has conditional property benefits, and the present causal model does not justify
adoption as the primary predictor. Track A semantics/convergence remain unresolved.

The [surrogate/property assessment](docs/surrogate_property_assessment.md)
explains what each completed stage establishes, compares property tradeoffs,
and works through reconstructed G, Sigma and occupation values. Track A target
convergence remains a standing limitation; the nominated SVO file contains DFT
and DC payloads, but their presence does not verify their physical alignment.

[Property visualization](docs/property_visualization.md) shows how to compare
one prepared material run to its existing exploratory model, and how to view
saved ED impurity spectra. Training-group views are explicitly labeled and do
not constitute an independent test or a material DOS/band reconstruction.

For supplied scalar AIM self-energy predictions, the
[reference-free property workflow](docs/aim_property_prediction.md) computes
impurity G and per-spin occupation without requiring ED targets or retraining.
It does not infer material conventions or predict a lattice DOS.

Use [the report-only commands](docs/closeout_reproduction.md) to regenerate
tables and all-state figures from saved results. Earlier training/evaluation
commands below document the study's development; they are not instructions to
reopen candidate search. Personal absolute source roots are excluded from the
new shared reports and the committed TRIQS inspection report.

## Research Objective

This project investigates physics-constrained latent
representation learning for electronic self-energy prediction
in strongly correlated materials.

The objective is to develop a computationally efficient
surrogate that complements DFT+DMFT calculations.

## Research workflow

Use the smallest experiment that can answer a stated question and change the
next decision. For exploratory work, keep input and target meanings explicit,
check the ED limiting cases or Track A source evidence, keep state groups and
partitions separate, and compare simple baselines. Investigate extra diagnostics
when they explain a result; publication-scale checks are not required after every
small run. Preserve a new untouched test for the final frozen comparison.

The fixed-bath AIM is a controlled reference, not a real-material calculation.
The original [Mitra and Banerjee study](https://doi.org/10.1038/s41524-026-02277-6)
already predicts self-energy from non-interacting Green functions; that mapping
alone is not a novelty claim. [Arsenault et al.](https://doi.org/10.1103/PhysRevB.90.155136)
examined AIM Green-function representations and training-set size. A materials
study of [out-of-distribution evaluation](https://doi.org/10.1038/s43246-024-00731-w)
shows why coverage and the intended holdout matter; it does not make a random
split invalid for an explicitly labeled interpolation question. Reproducibility
guidance such as the [NeurIPS program report](https://www.jmlr.org/papers/v22/20-303.html)
motivates recording settings and variability. These sources inform the workflow,
not universal numerical acceptance thresholds or a claim of novelty.
For connecting ED to DMFT later, finite-bath accuracy and bath-size effects need
their own assessment, as in [Liebsch and Ishida](https://arxiv.org/abs/1109.0158).

## Current Scope

The initial development focuses on Track A:

- Public DFT+DMFT dataset inspection
- HDF5 data extraction
- Physical and numerical validation
- Data provenance and metadata management
- Construction of verified training datasets

Track B separately provides a controlled finite-bath Anderson impurity reference.

### Working research direction

The immediate question is whether a compact self-energy representation preserves
Green functions and impurity properties as the environment changes, alongside
self-energy accuracy. The fixed-bath experiments
are controls, not the final task: epsilon-only interpolation has privileged
parameter access, and a one-parameter family does not establish broad latent
structure. Use full-target Ridge to distinguish the effect of target compression
from the effect of a linear predictor; use direct MLP to assess the supervised
nonlinear bottleneck under the same G0 inputs. Low-frequency error and physical
consistency accompany total Matsubara RMSE. Do not assume a smaller bottleneck
means a smaller model or faster inference.

The integrated study changes bath coupling strength and bath energies in two
separate experiments with matched epsilon_d values. The physical unit is an AIM
parameter tuple; all states at a
held-out bath scale stay together. Shared epsilon values are intentional controls,
not independent draws across baths. This measures transfer across coupling
or energy scales at familiar epsilon values, not generalization to arbitrary bath shapes,
temperatures or materials. A useful outcome is compressed prediction comparable
to full-target Ridge in Sigma, reconstructed G and impurity occupation. If good
Sigma RMSE coexists with poor property predictions, investigate low-frequency
error propagation and the training objective before increasing model complexity.

The next comparison is now implemented: matched refinement with a Sigma-only or
Sigma+G objective, described below. The research question is whether the training
objective and compression jointly preserve observables under bath transfer.
Keep data coverage fixed for this comparison. A fixed-pole causal decoder is now
available as the representation experiment; assess its consistency and accuracy under
both objectives before moving to impurity spectra. Vary U or extend materials
only when that resolves a specific limitation. These are research hypotheses,
not demonstrated general contributions. AIM representation learning is already
studied by [Arsenault et al.](https://doi.org/10.1103/PhysRevB.90.155136), and the
DFT-derived G0-to-self-energy task is established by
[Mitra and Banerjee](https://doi.org/10.1038/s41524-026-02277-6).
Novelty would require evidence for a specific advantage beyond those results.

Keep the two tracks connected through representations and diagnostics, not by
equating their inputs. DMFT involves an impurity mapping with self-consistency
([Georges et al.](https://doi.org/10.1103/RevModPhys.68.13)); this finite-bath AIM
has neither a DFT Hamiltonian nor a lattice self-consistency loop. Bath-size and
temperature accuracy in ED require their own analysis
([Liebsch and Ishida](https://arxiv.org/abs/1109.0158)). Track A still needs verified
input semantics and per-run convergence evidence. These limitations do not block
the small Track B experiment.

## Data Sources

The initial dataset is obtained from the public
Zenodo DMFT-ML dataset.

Original data files are stored locally and are not
included in this repository.

## Data Processing Principles

1. Preserve all original data.
2. Distinguish physical states from calculation runs.
3. Preserve DMFT iteration and spin-orbital information.
4. Preserve complex-valued arrays and original frequency meshes.
5. Do not assume the last iteration is converged.
6. Record missing metadata as unknown.
7. Separate extraction validation from physical convergence.

## Development Status

- HDF5 extraction and storage diagnostics support the observed TRIQS schema.
- Paired preparation and CPU baseline training are available through one CLI.
- Physical input semantics and per-run convergence require external evidence.
- A scalar fixed-pole causal decoder comparison is available; lattice spectral validation remains future work.
- Track B includes a small finite-temperature, two-bath AIM ED reference.

## Installation

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -e ".[dev]"
```

For baseline experiments, install `pip install -e ".[dev,ml]"`.

## Track B: finite-bath AIM reference

### Integrated representation-to-property workflow

One command now runs data generation, the existing candidate comparison, and
property reconstruction. It uses coupling scaling and energy scaling as two
separate cases: V_l=s V_l_base or epsilon_l=s epsilon_l_base, respectively. The
other bath parameters remain fixed. These are not a joint two-axis grid or
arbitrary bath shapes. U, beta and mu stay fixed, and paired epsilon_d values
control the comparison across bath scales. Default model seed is 31; specify
`--model-seeds 31 32 33` only when optimizer variation is the question.

```bash
conda activate physai
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m dmft_data.ed_workflow --output reports/local/aim_properties_v1
```

The output contains `recipe.json` (configuration and source hashes), one dataset
per axis, fitted models and splits, frequency plots, per-state validation property
arrays, and a combined `comparison.csv` / `summary.json`. Existing paths are never
overwritten. A failed later stage leaves earlier artifacts available. The workflow
always uses validation-only runs; the test scale remains reserved. All candidate
scores are retained. The selected model is still chosen by Sigma validation RMSE;
property columns are diagnostics and do not silently redefine selection.

New ED datasets store exact thermal `occupancy_per_spin`, computed directly from
the eigensystem. Predictions use G0 and predicted Sigma only:

```text
G_pred(iw) = [G0(iw)^(-1) - Sigma_pred(iw)]^(-1)
n_per_spin = 1/2 + (2/beta) sum_{n>=0} Re G(iw_n)
```

The density sum has an infinite-frequency tail. `ed_properties.py` reports the
finite sum and an approximation fitting Re G=-m1/w^2+m3/w^4 on the last 8-16
points and analytically summing those two missing terms. No true occupancy or
true moments enter that estimate. Error against the exact ED occupation is
reported for both predicted and reference G, so reference truncation/fit error
is visible separately from model propagation error. Reference accuracy does not
guarantee a learned tail is accurate; fitted-tail density is an approximation,
and out-of-range values are reported without clipping. All occupations are per
spin (total occupancy is twice this value). This follows the need to handle
tails in [Matsubara density calculations](https://triqs.github.io/triqs/4.0.x/doxygen/group__triqs-gfs-density.html),
but is a small independent NumPy implementation, not the TRIQS density algorithm.

For an existing candidate bundle from a new dataset, use
`python -m dmft_data.ed_validation --dataset DATA.npz --result RUN --output ANALYSIS --properties`.
Older datasets without ED occupations retain frequency-only analysis. The scalar
Dyson and occupation functions can also be used directly for new predictions;
the exact ED values are needed only for evaluation.

This is the first property stage, not spectral continuation. Predicted Matsubara
values cannot simply be substituted at real frequencies. Impurity spectral
prediction needs a justified real-frequency representation/continuation and an
ED reference. Lattice DOS/bands additionally need a lattice Hamiltonian, orbital
embedding/projectors, chemical potential and double-counting conventions, then
a lattice Dyson calculation. The AIM G0 here is a known finite-bath impurity
input, not a verified Track A DFT input or a self-consistent DMFT Weiss function.
No unimplemented spectral or lattice calculation is presented as a result.

### Decoder x loss comparison on the same saved data

Use the existing runner with `--decoder-comparison` to train four candidates:
latent-8 grid decoder / fixed-pole causal decoder, each with Sigma-only / Sigma+G
loss. No new ED data or separate evaluation pipeline is generated.

```bash
python -m dmft_data.ed_objective \
  --study reports/local/aim_properties_20261127_144210 \
  --output reports/local/decoder_comparison_v1 --decoder-comparison
```

This mode reuses the existing training-only scalers and (64,8,64) architecture,
but resets all network weights. Both heads receive identical initial hidden
weights. The grid head starts near the training mean Sigma; the causal head
near its positive-residue least-squares approximation. Each loss treatment
clones the corresponding start, with the same optimizer budget and regularizer.
This is not a comparison against a pretrained grid decoder. Original baselines
remain as context in the same output table.

The causal head predicts a real constant and softplus-positive residues:
`Sigma(z)=constant+sum_k residue_k/(z-pole_k)`. Default poles are 16 equally
spaced real values in [-4,4] in the AIM energy units. They are a prespecified
representation grid, not the two physical bath energies. `--pole-count` and
`--pole-bound` explicitly change that protocol; they are not fitted to validation.
Latent dimension 8 and pole count 16 are different quantities. Fixed poles
restrict representation capacity; this first experiment does not learn pole
positions. Positivity guarantees the scalar upper-half-plane imaginary sign,
not correct moments, occupation, real-frequency spectra or lattice DOS.

`metrics.json` includes `decoder_design`: head parameter counts, initialization,
and a training-only per-state bounded linear fit's physical Sigma RMSE and solver
success. That fit is a representation diagnostic, not a predictor and not a test
score. Output dimensions/initial predictions differ, so this comparison probes
the whole decoder parameterization, not positivity alone. Data, loss scaling,
hidden architecture, optimizer caps and held-out partitions remain shared.
No high-frequency moment penalty is added in this step. Read the four candidate
rows together; improved causality does not imply improved G or occupation.

### Resolve the measured pole-grid error before adding model complexity

The uniform 16-pole training fits on the seed-31 study concentrate about 98.5%
(coupling) / 98.8% (energy) of physical squared error at the first frequency.
This is an observed representation-fit diagnostic, not a validation prediction.
The first frequency is pi/20 ~= 0.157, while the nearest uniform poles are
+/-0.267. This motivates testing resolution near zero; it does not establish
where the true self-energy poles are.

`--pole-spacing low-frequency` keeps the pole count and endpoints fixed and uses
`p_k = w_min * sinh(t_k * asinh(bound/w_min))`, with equally spaced t_k in [-1,1].
Only the known frequency mesh determines this grid; no targets select positions.
The default remains `uniform` for reproducibility. Existing loss, architecture,
initialization policy and budgets are unchanged. This is a prescribed alternative,
not adaptive pole learning or an automatic validation search.

Use `--representation-only` with `--decoder-comparison` to inspect training-only
bounded fits without network optimization, checkpoint creation or held-out scoring:

```bash
python -m dmft_data.ed_objective \
  --study reports/local/aim_properties_20261127_144210 \
  --output reports/local/pole_resolution_low \
  --decoder-comparison --representation-only --pole-spacing low-frequency
```

`comparison.csv` reports training physical RMSE, standardized MSE and first-point/
first-eight error shares. Each `representation.json` contains per-frequency RMSE,
training indices, source hashes and solver status. These standardized-error fits
are not physical-RMSE lower bounds. On the existing seed-31 training data, the
alternative reduced physical fit RMSE from 0.05830 to 0.00261 (coupling) and
0.05251 to 0.00240 (energy); prediction improvement must be checked separately.

After that check, `--causal-only` trains only the two causal loss treatments.
This avoids repeating the unchanged grid decoder training; original saved
baselines remain in the comparison. Compare against the earlier uniform run
with identical source data, seeds and budgets.

```bash
python -m dmft_data.ed_objective \
  --study reports/local/aim_properties_20261127_144210 \
  --output reports/local/causal_low_frequency \
  --decoder-comparison --causal-only --pole-spacing low-frequency
```

### Historical closeout sequence

The sequence below has now been completed to the bounded scope described in
[the closeout report](docs/track_b_closeout.md), including saved impurity spectra.
It is retained as development history, not an active experiment queue.

The current milestone is a controlled Matsubara/occupation go/no-go assessment,
not demonstration of a universal DMFT surrogate. Finish it in a bounded sequence:

1. Resolve the observed representation issue with the prescribed grid alternative;
   retain a simple baseline, an unconstrained candidate and one causal candidate.
   Compare Sigma, low-frequency G, occupation and sign violations together. Do not
   demand that causality improve every metric or search indefinitely for a winner.
2. Fix the model/selection rule and optimization budget. Check a small declared
   set of initialization seeds for those finalists only. Keep budget termination
   distinct from convergence; random seeds are not independent physical states.
3. Evaluate frozen predictors on an unseen epsilon set within the same U, beta
   and bath family, with no fitting/scaling updates. If those results inform new
   choices, label them development data and preserve another final set. The
   uninspected scale-1.15 split remains usable for the narrower matched-epsilon
   bath-transfer claim. Frozen unseen-epsilon evaluation is now available below;
   the older fixed-bath external-test utility remains a separate fixed-bath tool.
4. Write the go/no-go result including capacity, budget, initialization variability
   and coverage limits. Lack of advantage over Ridge/PCA is a valid conclusion.

For the broader original Track B spectral objective, one further stage remains:
compare predicted impurity spectra against ED with the same broadening and energy
window, including spectral weight/occupation and high-frequency moments. Analytic
causality alone does not establish those properties; bulk Fermi-liquid constraints
should not be imposed indiscriminately on a finite bath. Completing the Matsubara
milestone does not complete spectral validation. Lattice self-consistency, crystal
DOS and real-material generalization are outside this controlled Track B closure.
Learnable pole locations, more bath sites and broad U/beta sweeps are follow-up
options, not prerequisites for finishing the stated milestone.

### Freeze the panel and stop adaptive candidate expansion

`ed_frozen` fixes three candidates from the original **uniform** decoder comparison:
PCA8+Ridge (baseline), reset latent8 grid with Sigma+G (unconstrained hypothesis),
and reset latent8 uniform-pole Sigma-only (causal control). This is a declared
development-informed panel, not three winners selected separately for each axis.
Low-frequency-pole runs remain representation ablations; they do not replace the
control automatically. No new composite loss, pole search or architecture is added.

Training policy is fixed at PCA8/Ridge alpha=1 and the existing reset latent8
procedure with L-BFGS-B max_iter=100, alpha=.001, G weight 1 or 0 respectively.
Budget-limited termination is preserved. PCA is an external baseline, not an
equal-parameter neural architecture. Freeze copies existing fitted models and
their scalers; it does not train anything. Available initialization seeds are
recorded from the source study. A seed-31-only source stays a single-seed result;
optimizer variability is not claimed resolved. If adding seeds later, declare
them and prepare the same finalists before inspecting the new evaluation.

```bash
python -m dmft_data.ed_frozen freeze \
  --study reports/local/aim_properties_20261127_144210 \
  --result reports/local/decoder_comparison_20261127_200055 \
  --output reports/local/frozen_panel_v1
python -m dmft_data.ed_frozen evaluate \
  --panel reports/local/frozen_panel_v1 --output reports/local/frozen_epsilon_v1
```

`protocol.json` fixes the candidates, source/checkpoint hashes, budget metadata,
evaluation seed (default 441), sample count (40), U, beta, mesh and bath scale
before evaluation. Evaluation creates one new dataset per axis at the original
validation scale (0.85), reusing that dataset across model seeds. It samples new
epsilon values from the same [-1,.2] range and rejects overlap with all original
epsilon values. It reports coverage relative to observed training extrema; a
few draws may lie outside those extrema despite sharing the generating range.
The reserved scale 1.15 is not opened. Evaluation checks the frozen bundle hash,
uses predict only, and saves all three models' scores and per-state predictions.
Original runs and frozen checkpoints are not overwritten. `--evaluation-seed`
and `--samples` are freeze-time options, not post-score tuning parameters.

Read Sigma RMSE, first-eight G RMSE and fitted-tail occupation RMSE together.
`dominates_pca8` means no worse on all three and strictly better on at least one;
it is a descriptive Pareto flag, not statistical significance or an automatic
model-selection rule. Failures remain in the table. No single winner or changed
hyperparameter is selected from these new scores. Once used to change a model,
the evaluation becomes development evidence, requiring another untouched final
set for a final claim. Repeated runs of the same seed are the same physical data,
not independent confirmations. This closes the no-refitting evaluation path;
multi-seed evidence, final scientific assessment and ED spectral comparison
remain distinct outstanding work.

### Candidate search is closed: prepare the three-seed final panel

The seed-441 unseen-epsilon result retained the Sigma/property trade-off: PCA8
had lowest Sigma RMSE, grid+G had lowest first-eight G RMSE on both axes, and
causal Sigma-only had lowest occupation RMSE on energy only. No candidate
dominated PCA8 on all three metrics. These results do not authorize another
architecture, pole or loss-weight search in the closing workflow.

The panel and budgets above are now closed. `prepare-final` preserves the frozen
seed31 checkpoints and trains only the two declared neural candidates at seeds
32 and 33. It reuses original training states/scalers, resets the hidden weights
with those seeds using the existing decoder initializer, and uses the same 100
L-BFGS-B iteration cap and alpha=.001. It exposes no architecture, seed-grid,
weight or budget tuning options. PCA8 is reused across seeds, not retrained.
It records actual optimizer termination and stops with a failure record if
preparation fails rather than silently dropping a seed. Neither validation nor
test targets enter training. Prior exploratory commands remain for provenance,
not as part of the closed-panel workflow.

```bash
python -m dmft_data.ed_frozen prepare-final \
  --panel reports/local/frozen_panel_v1 \
  --study reports/local/aim_properties_20261127_144210 \
  --result reports/local/decoder_comparison_20261127_200055 \
  --output reports/local/final_panel_v1
python -m dmft_data.ed_frozen evaluate \
  --panel reports/local/final_panel_v1 --output reports/local/final_reserved_v1
```

Preparation verifies that source dataset/bundle/metrics hashes match the frozen
panel, copies the original reserved partition without scoring it, and records
the new protocol and model hashes. Only the second command opens the scores.
Evaluation reuses predict/property analysis, verifies reserved-data hashes and
does not regenerate ED states, fit, pick a best seed or average model predictions.
It writes `comparison.csv`, per-state predictions and `seed_summary.csv` with
mean/population SD of each seed's RMSE and explicit successful-seed counts. This
is initialization variation on the same physical states, not a confidence
interval over independent physical samples. Failed evaluations remain visible.

This evaluation is labeled `frozen_reserved_scale`: scale 1.15 with the original
shared epsilon values. It is a different question from new epsilon at scale .85,
and neither is evidence of arbitrary-bath/material generalization. No automatic
winner is selected; report trade-offs against PCA8 under the fixed budget. The
single-seed scale-.85 result is already inspected development evidence. If the
reserved scores trigger further model changes, call that further work exploratory
and obtain a new untouched final set. After recording the three-seed reserved
result, close the Matsubara/occupation milestone with its limits; ED impurity
spectral reconstruction remains separate unfinished Track B work.

### After search closure: frozen impurity spectra against ED

The Matsubara comparison is closed. Keep PCA8 as the Sigma baseline, grid+G as
a conditional property-oriented alternative and the current causal model as a
control, without selecting new candidates from the reserved scores. The next
assessment uses the **same frozen causal model**, not a better-fitted spectral
model. It cannot establish a spectral ranking among all three candidates:
PCA/grid predictors have only Matsubara outputs, so they are explicitly marked
`unsupported` rather than subjected to an unvalidated continuation.

```bash
python -m dmft_data.ed_spectra \
  --panel reports/local/final_panel_v1 \
  --matsubara-result reports/local/final_reserved_v1 \
  --output reports/local/final_spectra_v1
```

`PoleBaseline.predict_z` evaluates the stored real poles and positive residues
at z=energy+i*eta, retaining the original Matsubara G0 input to the network.
`noninteracting_green_z` supplies the same finite-bath G0; Dyson reconstructs
the retarded impurity G. Existing ED solve/spectral functions provide references.
Each physical state is diagonalized once and reused over seeds and broadenings.
The default protocol uses [-8,8], 1601 grid points and eta=.05,.1 in AIM units.
These two specified broadenings describe sensitivity, not an optimization over
eta. Do not tune the grid/broadening to improve apparent agreement. Model,
protocol and reserved-data hashes are verified; no fitting or reselection occurs.

Outputs include per-seed spectrum RMSE and integrated absolute error, negative
weight fraction, finite-window spectral weight, and Fermi-weighted occupation.
Occupation is compared both to identically broadened ED and to exact thermal ED
occupation. The ED broadening/window bias is reported separately; spectra are
never renormalized or clipped to conceal missing weight. Sigma_infinity and the
1/z coefficient are obtained analytically from the learned constant and summed
residues and compared to U*n and U^2*n*(1-n), respectively, for this spin-symmetric
single-orbital AIM. No moment constraint is newly imposed on the model.

`comparison.csv` retains unsupported/failed rows. NPZ files contain all state
spectra and integrals; example PNGs show the first reserved state (index 0), not
a hand-picked best state. Passing `--matsubara-result` checks that the prior
report used the identical model/protocol/states and carries its seed summary
into the spectral summary for a consolidated record. Spectra remain broadened
finite-bath impurity functions, not lattice DOS. Accurate moments or causality
alone do not demonstrate spectral accuracy. If spectra are poor, report that
limitation rather than reopening the completed candidate search. Continuation
for the other candidates and application to real materials are separate future
work, not silently completed by this diagnostic.

The saved frozen seed31/32/33 predictions were checked without retuning: mean
integrated absolute spectral error was 0.981/0.988 (coupling/energy) at eta=.05
and 0.595/0.627 at eta=.1. At eta=.05 the mean finite-window weights were about
0.996 for both prediction and ED. Thus nearly matching total weight did not
imply accurate peak structure. These are conditional finite-bath results at the
stated broadenings, not a new model-selection criterion or evidence about the
unsupported PCA/grid spectra. The closed study supports reporting this causal
model's spectral limitation rather than reopening candidate search.

### Matched Sigma-only versus Sigma+G refinement

Reuse the completed integrated study; no new ED data or baseline training is
needed. The independent unit remains one AIM parameter tuple, with the same
held-out bath scale and shared epsilon values. The default comparison uses the
saved direct MLP and latent MLP with dimension 8. Each original checkpoint is
copied twice: one copy receives Sigma-only refinement and the other Sigma+G
refinement. Neither treatment starts from the other treatment's updated weights.
Unmodified nearest, Ridge, PCA and MLP candidates remain in the output table.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m dmft_data.ed_objective \
  --study reports/local/aim_properties_20261127_144210 \
  --output reports/local/green_objective_v1
```

The loss is `L_sigma + lambda * L_G + L2`, where:

- `L_sigma` is mean squared error over concatenated real/imaginary Sigma,
  standardized with the original training-only per-coordinate scaler. It is
  not the unweighted physical Sigma RMSE used in the evaluation table.
- `L_G = mean(abs(G_pred-G_ED)**2) / mean_train(abs(G_ED)**2)` is dimensionless.
  G_pred is computed with the scalar AIM Dyson relation. No validation or test
  values determine scaling, gradients, stopping or checkpoint choice.
- `L2 = alpha/(2*N_train) * sum(weights**2)` excludes biases; alpha is 0.001
  for both treatments. No occupation targets enter the loss.

Defaults are lambda=0 for the control and lambda=1 for the treatment, with 100
full-batch L-BFGS-B iterations and at most 2000 optimizer objective evaluations
each. Both use the same weights, frozen scalers, ReLU architecture and settings.
This is a matched **refinement** experiment, not a from-scratch comparison with
the original Adam training. NumPy implements analytic real/imaginary Dyson
derivatives and network backpropagation; SciPy runs the optimizer. Tests compare
the analytic derivatives to central finite differences through every layer.
No clipping, causal projection or analytic continuation is introduced.

`--green-weight` and `--max-iter` expose the protocol explicitly; a changed value
defines a new exploratory comparison. Do not search a large lambda grid simply
to improve this validation set. Actual iterations, optimizer evaluations,
termination messages, loss histories, versions, source/checkpoint hashes and
training-derived normalization are saved. Hitting the budget is reported as
`optimizer_success=false`, not convergence. Numerical failures are retained in
`failures.json`; successful but budget-limited predictors remain evaluable.

Read `comparison.csv` alongside the per-run `metrics.json`. Compare
`*_refine_sigma_g` with the **matching** `*_refine_sigma` row first, then compare
to original models and PCA 8/Ridge. Selection still uses Sigma validation RMSE;
it does not imply the best G or occupation predictor. Assess both bath cases:
better G and occupation at acceptable Sigma error supports retaining the joint
objective; inconsistent changes support a conditional conclusion, not a universal
advantage. No numerical acceptance threshold has been established for material
properties. Final test remains reserved until the comparison protocol is frozen.

G supervision here is determined by the same ED Sigma and known G0 through
Dyson. It changes the importance assigned to prediction errors; it does not add
independent information or prove causality. The proposed contribution is the
effect of this objective together with representation constraints on observable
preservation, subject to comparison with existing work.

Track A should be described separately. The
[source paper](https://doi.org/10.1038/s41524-026-02277-6) treats SrVO3, CaVO3,
BaVO3, SrNbO3 and BaNbO3 as formal d1 systems in ideal undistorted cubic
Pm-3m structures. This is the paper's calculation setup, not a statement that
all experimental phases or all local archived structures are identical.
Shared structure does not make bandwidth, interactions or orbital information
interchangeable. The current scalar two-bath AIM data contain no crystal
structure and are not material-specific d1 perovskite calculations. Connecting
this objective to Track A requires verified inputs/targets and the appropriate
orbital/lattice Dyson operation, not directly substituting a DFT local G0 into
the scalar impurity expression. See [the source evidence](docs/upstream_data_contract.md).

`dmft_data.ed` diagonalizes a spinful impurity and two spin-degenerate bath
sites in the 64-state Fock space. With orbital order
`(d_up, d_down, b0_up, b0_down, b1_up, b1_down)`, it diagonalizes `K=H-mu*N`:

```text
H = epsilon_d sum_sigma n_dsigma + U n_dup n_ddown
  + sum_l,sigma [epsilon_l n_lsigma + V_l (d_sigma^dagger c_lsigma + h.c.)]
G0(iw)^-1 = iw + mu - epsilon_d - sum_l V_l^2/(iw + mu - epsilon_l)
Sigma(iw) = G0(iw)^-1 - G_imp(iw)^-1
```

Use one common energy unit. `G_imp` is the thermal Lehmann Green's function,
spin averaged by default; no double-counting correction is applied. The
broadened `-Im G_imp(w+i eta)/pi` is an impurity spectral function. This
finite-bath reference has no DMFT self-consistency and is not a lattice DOS.

On Ubuntu, in `~/projects/dmft-self-energy`:

```bash
conda activate environment name
pip install -e ".[dev]"
pytest -q tests/test_ed.py
python -m dmft_data.ed --output data/processed/ed/aim_small.npz --samples 8 --seed 0
```

The local, Git-ignored NPZ contains the positive Matsubara mesh, parameters,
complex `green`, `g0`, `sigma`, and JSON metadata including the random seed and
NumPy version. Existing files are protected unless `--overwrite` is supplied.
The tests check analytic noninteracting and atomic limits, spectral
normalization, Matsubara causality and reproducibility. Finite bath size,
temperature and broadening limit the reference's physical scope.
The ED module also provides thermal impurity occupancy per spin. Its atomic-limit
checks use independent partition-function expressions; high-frequency tests use
the spin-symmetric moments `Sigma_inf = U*n_per_spin` and
`Sigma_1 = U^2*n_per_spin*(1-n_per_spin)`. These controlled AIM checks do not
establish the corresponding quantities or convergence in Track A data.
The generator varies U independently, while G0 does not contain U. A future
learning task must supply U (and other varying quantities not identifiable from
its input) or explicitly fix them; G0 alone does not define a unique Sigma
across independently varied U. The existing Track A blocked-partition helper can
be reused only after defining ED physical-state groups and parameter blocks;
the ED generator currently creates no train/validation/test split.

For an exploratory fixed-U CPU baseline, use a fresh local output path:

```bash
python -m dmft_data.ed --output data/processed/ed/fixed_u.npz --samples 60 --seed 0 --fixed-u 2.0 --beta 20
python -m dmft_data.ed_baselines --dataset data/processed/ed/fixed_u.npz --output reports/local/ed_fixed_u --seed 0
```

The second command holds out low and high impurity-level intervals for validation
and test, selects among fixed baseline settings on validation only, and saves the
selected model, split indices and metrics. This is an illustrative parameter
holdout, not evidence of generalization across arbitrary baths or materials.
Use fresh output paths for new experiments; do not retune on the reported test.

For exploratory comparisons, run the following from the repository root with
`physai` activated. These small example counts and ranges are starting points,
not a final assessment; keep a separate final test dataset for later research.
The same dataset is used for the random and epsilon splits, but their train and
test states differ, so their scores compare protocols rather than matched cases.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# A: independent-state interpolation versus epsilon-blocked extrapolation
python -m dmft_data.ed --output data/processed/ed/compare_a.npz --samples 100 --seed 13 --fixed-u 2 --beta 20
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_a.npz --output reports/local/aim_a_random --split random --seed 13
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_a.npz --output reports/local/aim_a_blocked --split epsilon --seed 13
# B: repeat the split comparison on a fixed bath; change the bath for another case
python -m dmft_data.ed --output data/processed/ed/compare_b.npz --samples 100 --seed 13 --fixed-u 2 --beta 20 --fixed-bath -1 1 0.5 0.7
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_b.npz --output reports/local/aim_b_random --split random --seed 13
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_b.npz --output reports/local/aim_b_blocked --split epsilon --seed 13
# C: U varies; it is explicitly appended to G0 as an input feature.
python -m dmft_data.ed --output data/processed/ed/compare_c.npz --samples 100 --seed 13 --beta 20
python -m dmft_data.ed_baselines --dataset data/processed/ed/compare_c.npz --output reports/local/aim_c_u --split u --seed 13
python -m dmft_data.ed_plot --result reports/local/aim_c_u
```

`--split u` reserves U in [1.5,2) for validation and [2,2.5) for test;
remaining sampled U values are training. These are interior holdouts, not
outside-range extrapolation. The PNG shows individual test-state errors rather
than smoothing or averaging over baths, and the worst low-frequency test curve.
Each run saves predictions and reference values for detailed inspection. ED
parameters, fixed bath and sampling ranges are retained in the local dataset;
metrics include split indices, seed, model settings and fit warnings. All test
figures and metrics are exploratory once examined; do not use them as untouched
final evaluation. The chosen fixed bath and intervals are illustrative.

### Latent comparison (exploratory)

The fixed-bath interpolation run found nearest neighbor competitive, while the
blocked run was harder. This motivates comparing compression and a nonlinear
bottleneck against the same baselines before adding a causal decoder. It does
not establish a generalization or novelty claim.

### Controlled bath-strength comparison

`--bath-scales` uses `--fixed-bath` as a base: energies stay fixed and each
coupling is multiplied by a positive scale s, so Delta_s(z) = s^2 Delta_base(z).
U must be fixed. `--samples` is the number of epsilon_d values **per scale**;
the same draws appear at every scale. The example generates 200 states at five
scales, trains on 120, validates on 40 at s=0.85 and reserves 40 at s=1.15.
Both held-out scales are interior to the training scale range [0.7,1.3]; this is
interpolation in that parameter, not parameter-range extrapolation. Input-space
coverage is a separate question, as illustrated in
[materials OOD research](https://arxiv.org/abs/2406.06489).

```bash
python -m dmft_data.ed --output data/processed/ed/bath_transfer_seed31.npz \
  --samples 40 --seed 31 --fixed-u 2 --beta 20 --fixed-bath -1 1 0.5 0.7 \
  --bath-scales 0.7 0.85 1.0 1.15 1.3
for s in 31 32 33; do
  python -m dmft_data.ed_baselines \
    --dataset data/processed/ed/bath_transfer_seed31.npz \
    --output "reports/local/bath_transfer_seed${s}" --split bath \
    --validation-bath-scale 0.85 --test-bath-scale 1.15 \
    --seed 31 --model-seed "$s" --include-latent --include-ridge \
    --validation-only --save-candidates || break
done
python -m dmft_data.ed_validation --dataset data/processed/ed/bath_transfer_seed31.npz \
  --result reports/local/bath_transfer_seed31 --output reports/local/bath_transfer_frequency
```

`--include-ridge` adds alpha=1 full-target Ridge with the same training scalers
and regularization as PCA+Ridge. No tuning is performed for these baselines.
All candidates receive G0 only; scale labels are used for splitting, not as
features. The fixed-bath epsilon-only baseline is invalid here and is rejected.
Seeds 31/32/33 vary initialization on the same states, not physical replication.
Explicit bath holdouts do not depend on the split seed. Existing candidate
checkpointing and frequency diagnostics are reused. External-test evaluation
currently supports only fixed-bath datasets; this experiment uses the in-dataset
bath split and leaves the reserved targets unscored in validation-only mode.

### Fixed-bath latent example

Use a fresh dataset and output directory, for example:

```bash
python -m dmft_data.ed --output data/processed/ed/latent_example.npz --samples 100 --seed 31 --fixed-u 2 --beta 20 --fixed-bath -1 1 0.5 0.7
python -m dmft_data.ed_baselines --dataset data/processed/ed/latent_example.npz --output reports/local/aim_latent_random --split random --seed 31 --include-latent
```

`--include-latent` compares PCA+Ridge and unconstrained deterministic MLPs
with 1, 2, 4, 8 and 16 bottleneck units alongside the mean, nearest and direct MLP.
The MLP uses ReLU layers of widths 64, latent dimension, 64 with linear output;
training-only standardization and a 500-iteration optimizer budget match the
direct MLP. PCA is target-side compression, not a learned input encoder. Its
dimensions are not silently clipped: at least 17 training states and 16 real
target features are required for dimension 16.
Each candidate's validation errors, sign-violation fraction, seed and fit
warnings are saved; only the validation winner is evaluated on test. The last
8-point metric is a finite-window error, not a high-frequency moment test.
The MLP decoder is unconstrained: sign checks on sampled Matsubara points do not
prove causality. Causal decoders and real-frequency property reconstruction
remain subsequent work, not part of this comparison. Preserve a separate final
test before making research claims; previously inspected tests are exploratory.

For a targeted training-stability question, use `--compare-activations` with
`--include-latent --validation-only` and repeat a prespecified set of
`--model-seed` values into separate directories. This adds matched tanh variants
of the direct and latent MLPs; tanh applies to **all hidden layers**, not only
the bottleneck. The validation-only option does not predict or score reserved
test samples, but it still reads the local dataset. Model initialization seeds
on the same physical states are not independent samples. The output records
activation, stopping settings, warnings and train/validation bottleneck behavior;
tanh saturation is reported instead of calling negative tanh activations dead.

```bash
for s in 31 32 33; do
  python -m dmft_data.ed_baselines --dataset data/processed/ed/latent_example.npz \
    --output "reports/local/activation_seed_${s}" --split epsilon --seed 31 \
    --model-seed "$s" --include-latent --compare-activations --validation-only || break
done
```

For the fixed-bath question, `--include-epsilon-linear` adds a baseline using
**epsilon_d directly**. The other candidates use G0, so this is a comparison
with different input information. It requires fixed U, bath parameters, mu and
beta across the dataset and at least two distinct training epsilon values;
duplicate training epsilon values are rejected. It fits only training targets,
linearly interpolates each real and imaginary component, and extrapolates with
the slope of the nearest endpoint pair. The latter is a deliberately simple,
potentially unstable baseline for the blocked split, not a physical guarantee.
Metrics record the training range and validation counts below, within and above
that range; test coverage is recorded only when test evaluation is enabled.

```bash
python -m dmft_data.ed_baselines --dataset data/processed/ed/latent_example.npz \
  --output reports/local/aim_linear_validation --split epsilon --seed 31 \
  --model-seed 31 --include-latent --include-epsilon-linear --validation-only
```

The reserved high-epsilon block from an already inspected dataset remains
exploratory. Before any final evaluation, freeze the candidate selection,
training and evaluation protocol and reserve independently generated physical
states. A saved validation winner can be evaluated once with:

```bash
python -m dmft_data.ed_external_test \
  --development data/processed/ed/latent_example.npz \
  --result reports/local/aim_linear_validation \
  --test data/processed/ed/independent_final.npz \
  --output reports/local/independent_final_evaluation
```

Use only trusted local joblib artifacts. The evaluator verifies the development
dataset hash, fixed parameters, exact frequency mesh and distinct physical
states; it cannot establish that a dataset was never previously examined. It
uses the saved validation winner without refitting and reports errors separately
below, within, and above its training epsilon range. For a prespecified comparison,
add `--save-candidates` to a new validation-only run, then add `--all-candidates`
to the external evaluation command. All saved candidates are scored on the same
states without refitting; the bundle hash is checked. Older runs save only the
winner. Freeze models, training split, test sampling and primary metrics before
examining a genuinely independent test; the program cannot prove a test is untouched.
For a frequency-resolved exploratory comparison of a saved validation bundle,
use a result made with `--save-candidates`:

```bash
python -m dmft_data.ed_validation --dataset data/processed/ed/latent_example.npz \
  --result reports/local/aim_linear_validation --output reports/local/aim_frequency
```

This saves a CSV and figure for validation states only, with complex, real and
imaginary errors and signed mean errors.

The epsilon baseline has different input information. Treat all final scores as
final evaluation, not a further tuning set. Only load trusted joblib artifacts.
No interpolation or extrapolation result establishes lattice DOS accuracy or
causality, and extrapolated values need not obey the Matsubara sign constraint.

Remaining research work includes matched independent fixed-bath comparisons and
low-frequency assessment; variation across bath parameters and interaction with
identifiable inputs; a justified causal decoder against these baselines; and
independent impurity spectral evaluation before any lattice property analysis.
Track A still requires source input semantics and per-run convergence evidence.
These are research questions, not results or claims of novelty.


Inspect the per-candidate validation errors, train loss and bottleneck metrics;
keep all seeds and identify optimization failures. The same 500-iteration limit
does not equalize wall-clock work or model capacity. The old default also scores
the exploratory test; use `--validation-only` while developing, and freeze a
protocol with a fresh independent test before a final claim. Current sign checks
are sampled necessary conditions, not proof of causality; spectral properties
and lattice DOS remain later, separate analyses.

The fixed-bath example is a one-parameter family, but a one-dimensional nonlinear
curve need not be well represented by one linear PCA component. The exploratory
blocked run showed a real offset and low-frequency sign violations; therefore
`diagnostics.json` records per-test-state oracle real offsets, shape residuals and
individual positive-imaginary points. Offsets use true targets and are not a
deployable correction. Training-only PCA explained variance and reconstruction
errors on train and validation are diagnostic, not model selection scores.

Use `--model-seed` to vary optimizer initialization while holding `--seed` and
the random partition fixed; both are recorded in `metrics.json`. With an epsilon
blocked split the thresholds are fixed, and changing split seed does not change
its partition. Repeat runs into fresh directories, retaining seed-level results;
the same maximum MLP iteration count does not imply equal computational budgets
or equal tuning. Validation and test data must never be used for fitting scalers
or PCA. These diagnostics do not prove causality or establish the self-energy
high-frequency moment, and they do not assess lattice DOS.

`diagnostics.json` also records each MLP's training loss curve and convergence
settings. The loss is scikit-learn's regularized objective on standardized
targets, not the complex RMSE in native units; iteration counts alone do not
identify why optimization stopped. For each latent MLP, the train and validation
summaries count bottleneck units that were never positive, samples with all-zero
bottlenecks, and activation variability. "Never positive" is an observation on
the specified samples, not proof that a unit can never activate. These diagnostics
do not use test inputs or targets and do not establish physical meaning of a
latent coordinate. Compare parameter counts as well as bottleneck dimensions:
a narrow representation does not imply fewer trainable parameters. Model seeds
on one fixed split are repeated initializations, not independent physical states.

## Recommended learning workflow

On the Ubuntu machine, activate `environment` and work in
`~/projects/dmft-self-energy`:

```bash
python -m dmft_data catalog --source data/raw/hdf5 --output data/metadata/run_review.json
# Inspect the catalog and set input/target paths in configs/experiment.yaml.
python -m dmft_data prepare --config configs/experiment.yaml
python -m dmft_data summary --config configs/experiment.yaml
python -m dmft_data train --config configs/experiment.yaml
```

The example runs an explicitly exploratory stored-array experiment, even while
convergence remains `not_assessed`. It keeps spin/orbital blocks together and
groups replicas before fitting. Primary experiments use reviewed physical-state
IDs, scoped convergence evidence, and verified input semantics. See
[the workflow and evaluation protocol](docs/learning_workflow.md) for review
format, grouping, model selection, output artifacts, and limitations.

After training, `python -m dmft_data diagnose --config configs/experiment.yaml`
bundles physics definitions, signed block errors and plots without retraining.
See [physics definitions and diagnostics](docs/physics_diagnostics.md) for the
review template and optional, explicitly selected raw-run/Dyson checks.
The [physics validation protocol](docs/physics_validation_protocol.md) sets out
the next reference, convergence and observable checks. Diagnostics include
block-resolved PNG/PDF figures, caption drafts and a machine-readable check plan.

## Data Processing

The local data directory is configured through
`configs/dataset.yaml`.

The earlier normalized-HDF5/compatibility export workflow remains available:

```bash
python scripts/01_inventory.py
python scripts/02_extract.py
python scripts/03_validate.py
python scripts/04_build_dataset.py
```

For a local laptop with large or restricted HDF5 data, create a small transfer
bundle without sending the source payload:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/06_collect_laptop_results.py \
  --source /path/to/local/hdf5 \
  --output laptop-results
```

Send the generated `laptop-results/laptop_collection.zip` back to the
workspace. The ZIP contains inventory, extraction, numerical-validation, and
TRIQS-style diagnostic reports; it does not contain original HDF5 payloads.
The collector replaces the supplied local source-root prefix with
`<LOCAL_SOURCE_ROOT>`, but review the bundle before sharing because HDF5
attributes may contain other project-specific information.

After local normalized-data processing and `04_build_dataset.py`, record a
fail-closed ML admission decision:

```bash
python scripts/09_ml_readiness.py \
  --dataset-build-report reports/research_dataset_build.json \
  --output reports/local/ml_readiness.json
```

`ready` only means the dataset-build report recorded explicitly verified
converged records and a non-overlapping material-blocked holdout. It is not a
claim that this script independently validated the HDF5 data, convergence,
causality, or spectral reconstruction.

It is not required before the paired baseline workflow above. Legacy convergence
registries now require per-run SHA-256, exact `verified_source_paths`, a
`verified_converged` status, and an evidence reference. Collection-wide booleans
and unreviewed HDF5 flags cannot certify primary targets.

### HDF5 storage diagnostics

`python -m dmft_data storage --source data/raw/hdf5 --output reports/local/run_diagnostics`
compares final `G_iw` and `Sigma_iw`
blocks with every stored iteration. It reports storage equality and numerical
differences, including a configurable low-positive-Matsubara-frequency RMS
(`--low-frequency-count`, default 10). It is a storage/consistency diagnostic:
it does not establish DMFT convergence, and all outputs retain
`convergence_status: "not_assessed"` and `primary_target_candidate: false`.
Paths under `beta-variation` are only marked `parameter_variation_candidate`
pending metadata review. Warmup settings, iteration records, and spin/orbital
blocks are not independent physical states.
The older `scripts/07_classify_hdf5_runs.py` command remains available and uses
the same implementation. An absent source directory fails clearly; an empty
directory produces a warning rather than silently looking like a useful scan.

For a fail-closed raw-data provenance audit and target/split registry, see
[research-readiness audit](docs/research_audit.md). This creates local JSON
reports only; without independently supplied convergence evidence it selects
no primary ML targets.

## Reproducibility

All extraction settings, source checksums,
validation results, and dataset split definitions
should be preserved.

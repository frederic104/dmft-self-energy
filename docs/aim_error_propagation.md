# Post-hoc self-energy error propagation in the AIM

Question: for the existing frozen AIM predictions, how do the finite-window
real-average Sigma error and the remaining frequency-dependent complex error
affect impurity G and per-spin n? The comparison baseline is the full saved
prediction, with PCA8, grid Sigma+G and causal Sigma under their existing splits.
One physical unit is an AIM parameter tuple. Seeds vary model initialization on
the same states; they are not independent physical samples. This descriptive
study informs which errors matter, not model selection. Candidate search remains
closed, and BVO/material semantics are not resolved by AIM results.

## Definitions

For each state on the existing 64 positive Matsubara frequencies, let
`delta = Sigma_pred - Sigma_ED`, `c = mean_omega Re(delta)`, and `r = delta - c`.
`c` is a real finite-window average, **not an established Hartree term or
high-frequency moment**; `r` includes all of the imaginary error. Each of the
three cases uses the same known AIM noninteracting impurity propagator G0:

| Case | Sigma passed through the existing Dyson reconstruction |
| --- | --- |
| full | Sigma_pred |
| real_offset_only | Sigma_ED + c |
| residual_only | Sigma_ED + r |

The two synthetic cases explicitly use ED reference Sigma. This is an oracle
diagnostic, not a usable prediction correction, physical intervention, new
test, or a causal attribution to the training procedure. Counterfactual inputs
are not guaranteed to be physical self-energies. G uses the exact nonlinear
scalar Dyson equation and n uses the existing finite-mesh tail estimate. G RMSE
is over the first 8 points; n RMSE is against exact thermal ED occupation per
spin. The full case must reproduce the saved scores. The code also requires ED
G0/Sigma/G to satisfy Dyson numerically; failures are recorded, not replaced by
zeros. Reference n-tail error is separate from model error.

The exact nonlinear interaction is
`I_G = G_full - G_offset - G_residual + G_ED` and similarly
`I_n = n_full - n_offset - n_residual + n_ED_tail`, with the same tail estimator
for all n terms. Thus signed changes add only after including interaction. RMSEs
are norms and **do not add**, even if the interaction is small. They include
cross-state and frequency cancellations differently from signed contributions.

## Executed result: reserved scale

The following are **means of seed-level RMSEs** on the same 40 states per axis,
seeds 31/32/33. PCA8 is deterministic and repeated; its three rows are not three
physical samples. Scale 1.15 is held out from fitting but lies within the range
of training scales, and does not establish extrapolation or material OOD ability.
Interaction columns are the same for all three case rows and are shown once.

| Axis | Model | G RMSE full / offset / residual | n RMSE full / offset / residual | G interaction RMSE | n interaction RMS |
| --- | --- | --- | --- | ---: | ---: |
| coupling | PCA8 | .082703 / .014376 / .077276 | .014427 / .006679 / .011724 | .004712 | .001147 |
| coupling | grid Sigma+G | .086566 / .014340 / .086971 | .014187 / .006781 / .014470 | .003529 | .000873 |
| coupling | causal Sigma | .218738 / .009850 / .220337 | .049010 / .005836 / .049913 | .008252 | .001447 |
| energy | PCA8 | .101919 / .007969 / .099714 | .018359 / .003520 / .017297 | .003096 | .000568 |
| energy | grid Sigma+G | .070507 / .024268 / .074101 | .014362 / .010250 / .012798 | .005172 | .000843 |
| energy | causal Sigma | .284098 / .013289 / .278682 | .064855 / .005158 / .061491 | .007754 | .001816 |

The residual-only G RMSE is much larger than offset-only in these cases; that
does not mean the offset has no effect on n, and norm comparisons cannot uniquely
attribute a share of the full error. Energy grid n has substantial sensitivity
to both cases. In seed 31 its full/offset/residual n RMSE is
.008694/.012314/.010387, illustrating cancellation; see the CSV for every seed
and state. These descriptive conclusions are limited to the stored AIM family.
Track B verifies the analysis for its defined AIM G0; it cannot prove whether
an archived material `G0` is a Weiss function.

## Reproduce from saved arrays

```bash
conda activate physai
cd ~/projects/dmft-self-energy
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m dmft_data.ed_report \
  --results reports/local/frozen_epsilon_v1 reports/local/final_reserved_v1 \
  --output reports/local/aim_error_propagation_v1
```

Choose a new output directory for each run. `counterfactual_metrics.csv` has
one row per run/axis/model/seed/case and retains failed cases with reasons;
`counterfactual_states.csv` preserves every state and signed n interaction.
The same interaction is repeated across case rows; do not sum those rows.
`summary.json` identifies source/code hashes, missing inputs and failed
counterfactuals. All phases remain separate: unseen-epsilon uses only seed 31
and is not pooled with reserved-scale results. Model selection, ED generation,
checkpoint loading, inference, fitting and analytic continuation are not run.

Executed on existing copied saved results: two runs, 40 states per axis per run,
72 per-seed case rows and 2,880 case/state rows; zero missing inputs and zero
counterfactual failures. Saved reference arrays, dataset checksums and full
predictions were checked by the existing reporter. These row counts include
seeds and counterfactual cases and are not independent observations. Analytic
fixtures check the nonlinear interaction and reject inconsistent Dyson inputs;
tests do not reproduce original data generation or model training.

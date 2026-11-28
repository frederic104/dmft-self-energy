# Reproduce the closed-study report

Run from the updated repository on Ubuntu. The commands use the existing physai
environment and the result directories verified during the closeout. Personal
absolute roots are intentionally absent from this document.

```bash
conda activate physai
cd ~/projects/dmft-self-energy
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

python -m dmft_data.ed_report \
  --results \
    reports/local/aim_properties_20261127_144210 \
    reports/local/green_objective_20261127_191941 \
    reports/local/decoder_comparison_20261127_200055 \
    reports/local/causal_low_20261127_203835 \
    reports/local/pole_resolution_20261127_203835 \
    reports/local/frozen_epsilon_v1 \
    reports/local/final_reserved_v1 \
    reports/local/final_spectra_v1 \
    reports/local/linear_frequency_random_seed31 \
    reports/local/linear_frequency_blocked_seed31 \
  --output reports/local/track_b_closeout_v2
```

This reads existing results only. Do not run ed_workflow, ed_objective or
ed_frozen prepare-final to recreate this report: those commands train models.
No model inference, ED data generation, continuation, selection or tuning occurs.
The output path must be new. Change only the report output directory to repeat.

For only the final results, pass final_reserved_v1 and final_spectra_v1 to
the same --results option and choose another new output directory. Phase labels
are retained; results from different stages are never pooled into one score.

Expected outputs:

| File | Meaning |
| --- | --- |
| report.md | Tables separated by source run, phase, axis, candidate and eta |
| comparisons.csv | Every original candidate row, including unsupported/failed records |
| metric_summary.csv | Mean and population SD of seed metrics, successful/expected counts; single-seed SD empty |
| training_termination.csv | Recorded optimizer stops; no optimization is performed |
| state_errors.csv | All frozen states/seeds; reference/predicted G and Sigma at omega_0; raw/tail-corrected n, exact ED n, errors and signed occupation decomposition |
| paired_state_errors.csv | Same-state/same-seed squared-error differences relative to PCA8 |
| frequency_errors.csv | Sigma/G/linearization errors, exact amplification and expansion parameter |
| dyson_diagnostics.csv | Exact identity residual and relative linear remainder for every case |
| spectral_state_errors.csv | All-state spectral RMSE and integrated absolute error |
| states_00.png through states_07.png | All-state plots for the full command above; spectral ECDFs included |
| summary.json | Relative artifact labels/hashes, source hashes, versions, missing-input list and diagnostic summaries |

Plots average seed diagnostics at each state, not predictions. All seed-level
rows remain in CSV. Existing summaries without frozen arrays still contribute
their original tables. Missing frozen inputs are listed explicitly; no scores
or arrays are invented. A supplied dataset with a wrong recorded hash, mismatched
references, or inconsistent saved scores causes an error.

For the subsequent [surrogate/property assessment](surrogate_property_assessment.md),
use the same command with `--output reports/local/surrogate_property_assessment_v1`.
The extended CSV reports signed `occupancy_first_8_contribution`,
`occupancy_remaining_contribution`, `occupancy_tail_correction_difference` and
`reference_tail_minus_exact`; their sum equals `occupancy_error`. The first two
terms use predicted minus reference Re G, multiplied by 2/beta. The tail term
compares the two fitted corrections, not the full corrected occupations.
`reference_occupancy_raw/tail` and `predicted_occupancy_raw/tail` are per spin;
`omega_0` and the `*_iw0_real/imag` columns identify actual complex values.
Do not add component RMSEs or pool seed rows as independent physical states.

Frozen evaluators already save all arrays needed here. No extra inference was
needed for the available results. If a prediction NPZ is lost, restore it from
the existing result first. Only if unavailable, an explicit no-fit evaluation
of the same trusted frozen checkpoint into a new directory could recreate it;
that would be additional inference and must not be described as a new test.

Analytic/property and report tests, without training experiment fixtures:

```bash
python -m pytest -q \
  tests/test_ed_report.py tests/test_ed_spectra.py tests/test_ed.py \
  tests/test_ed_workflow.py::test_density_tail_matches_noninteracting_fermi_function \
  tests/test_ed_workflow.py::test_interacting_atomic_density_and_dyson_error_propagation \
  tests/test_triqs_inspection.py tests/test_laptop_collection.py \
  tests/test_run_diagnostics.py tests/test_research_audit.py
```

The new tests use analytic saved-array fixtures and forbid model loading,
prediction and fitting. Existing ED tests exercise limiting cases and a tiny
spectral fixture. Test success is not independent verification of the historical
training or material data semantics.

## Checks executed during this closeout

Implementation scope:

- ed_properties.dyson_errors adds exact/linear scalar error decomposition.
- ed_report.report consolidates existing files; frozen_diagnostics and
  spectral_diagnostics reuse existing property metrics, and aggregate preserves
  seed-level statistics. It exports all-state, frequency and paired errors.
- provenance.redact_source_root centralizes the collector's existing scoped
  redaction and is reused by TRIQS inspection. Report producers suppress roots;
  operational data readers still receive their paths through existing CLI/config.
- README and research documentation connect the closed results to the unresolved
  material contract and bounded property workflow. No model architecture or
  training/evaluation selection rule changed.

The command above passed **30 tests** on the local Windows Python 3.12.7
environment (NumPy 2.0.0, SciPy 1.16.0, scikit-learn 1.8.0, Matplotlib 3.9.2,
pytest 7.4.4). An initial subprocess test could not import dmft_data because
PYTHONPATH had not been exported; exporting src as above resolved it, and the
complete listed selection then passed. This was not an Ubuntu test run or a
recreation of the original training environment. Ruff was unavailable and was
not run; compile checks and git diff --check passed.

The full report command was executed against copies of the existing Ubuntu
results: 34 input artifacts read, zero missing inputs, 18 output files including
eight PNGs. Saved frozen Matsubara and spectral metrics agreed with recomputation.
Both state-error and spectral-distribution figures were visually inspected.
All 62 source-file hashes in the redacted tracked TRIQS report were preserved.
Tracked text files and the new report's JSON/CSV/Markdown were checked for
personal home-directory roots; none remained. README/docs were checked for
unintended Korean text; repository documentation remains English.

The saved final-panel preparation metadata records Python 3.13.16, NumPy 2.5.2,
SciPy 1.18.1 and scikit-learn 1.9.1. These are recorded historical versions, not
packages reinstalled or training runs repeated during this closeout. The report
records its own runtime versions and the hashes of all source reports.

## Privacy of shared reports

New closeout artifact keys use run-name/relative-file labels. TRIQS inspection
uses <LOCAL_SOURCE_ROOT>, retaining file SHA-256 and relative source paths.
The collector reuses the same scoped redaction, including source-root strings
inside nested attributes/errors. Storage/audit summaries hide their root, and
new objective recipes record the study label plus its hash instead of an absolute
path. HDF5-internal paths remain meaningful and unchanged.

Existing local experiment originals are preserved because rewriting their
metadata would change provenance hashes. Legacy operational extraction outputs
may still contain paths needed locally; they are not included in this shareable
closeout. Arbitrary metadata outside the declared source root is not guaranteed
anonymous. Already-published Git history is not rewritten by this change.

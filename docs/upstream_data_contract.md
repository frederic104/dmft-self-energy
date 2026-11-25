# Source evidence and the next experiment

Reviewed 2026-11-24. This note records shared source evidence once; it does not
require another scan of all local HDF5 files. Publication statements, public
preprocessing, and local observations are different evidence levels.

## What the public sources establish

[Mitra and Banerjee (2026)](https://doi.org/10.1038/s41524-026-02277-6)
describe predicting self-energy from DFT-derived non-interacting Green functions,
cubic perovskites, a 15-iteration DMFT procedure, and warmup-conditioned curves.
The paper links the [data record](https://zenodo.org/records/17215589) and
[public code](https://github.com/hrishitbanerjee/Deep-DMFT). These are shared
literature statements, not independently verified convergence certificates for
each local file. The data-record contents were inaccessible during this review;
no archive was downloaded.

### Calculation software: checked scope

The paper's Methods identify VASP for structural relaxation, TRIQS/DFTTools
with projective Wannier orbitals for DMFT, and TRIQS/CTHYB for the impurity
solver. It describes a fully localized limit double-counting correction and a
rotationally invariant Kanamori interaction, with U in a material-dependent
4–5 eV range and J_H = 0.5 eV. These are publication-level statements;
the exact U for each local run, software versions, stage-specific array meanings,
and applied double-counting values still require source-specific evidence.

[DFTTools' formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html)
distinguishes the lattice projection and auxiliary impurity problem;
[CTHYB's moment guide](https://triqs.github.io/cthyb/latest/guide/high_freq_moments.html)
describes high-frequency moments derived from the model. Neither documentation
proves what a particular archived `G0-i` contains. The
[VASP EDIFF definition](https://vasp.at/wiki/index.php/EDIFF) concerns electronic
self-consistency during VASP calculations; it is not an accuracy threshold for
predicted self-energy. `solid_dmft` documentation is a useful comparison for
possible output observables, but the paper does not establish that application
or its output schema as the producer of these files.

The public
[preprocessing notebook](https://github.com/hrishitbanerjee/Deep-DMFT/blob/main/01_preprocess_svo.ipynb),
Git blob `2df768e8e706bee027114f495504003b84efc71d`, code cell
`cc_0504b531`, function `readfile`, explicitly:

- reads `dmft_output/G0-i` and `dmft_output/Sigma-i` for the **same** `i`;
- loops over 25 iterations and six spin/orbital labels;
- creates separate channel/iteration rows and repeats the warmup count;
- selects a window using `linspace(-161, 161, 2050)`, rather than reconstructing
  the stored Matsubara mesh.

`Sigma_iw` is accessed in that function but is not the training-target payload.
This is a reader, not the DMFT generation code: it does not establish how `G0-i`
was constructed or whether its index denotes solver input before/after mixing.
The 25-iteration notebook and 15-iteration paper/local archives must not be
silently reconciled. A special reader in the notebook is explicitly described
for LVO; it does not explain BVO's final=iteration-5 storage.

## Meaning of the present baseline

| Selection | Supported interpretation |
| --- | --- |
| `G0-1 → Sigma_iw` (current default) | Initial stored-array to archive-final mapping; exploratory |
| `G0-k → Sigma-k` | Same-index stored-array mapping; closer to the public preprocessing pairing |
| Verified DFT G0 → verified converged Sigma | Intended material-surrogate task; requires the corresponding evidence |

These tasks answer different questions. Do not relabel the existing nearest
neighbor score as a reproduction of the published experiment or as a verified
impurity solver. Same-index pairing alone does not prove solver-stage alignment.
We retain actual complex meshes and all channels in each run, and keep replicas
together in blocked splits rather than copying the public row-level sampling.

The user-reported BVO files have archive-final arrays equal to iteration 5, and
one inspected file has `/dmft_output/iterations = 5` despite numbered groups up
to 15. This supports investigating archive bookkeeping; it does not establish
which numbered groups belong to the intended trajectory or physical convergence.
No stored target is promoted to converged by this note.

## A bounded next comparison

1. Preserve `exploratory_v1` and its configuration. The BVO test has already been
   inspected; subsequent revisions are exploratory, not a fresh blind test.
2. Define one same-index comparison at a **predeclared fixed index**. Index 1 is
   a proposed starting comparison if both paths exist; it is not a converged-target claim.
   Keep the split, features, models, seeds, and frequency count unchanged.
3. Use separate prepared/experiment directories. Review per-run selectors in
   `run_review.json`: they override config selectors. `summary` now prints actual
   quantity pairs from prepared records, including overrides.
4. Compare validation errors and training-only PCA reconstruction before deciding
   whether a learned latent/causal decoder is useful. Keep iteration tasks separate
   from final-target tasks; do not pool them as extra independent states.
5. For physical claims, resolve only the decisive missing definitions: G0
   construction/stage, Sigma/DC and energy conventions, material parameter
   mapping, and target convergence evidence. A shared authoritative generation
   source can resolve common settings; repeated raw-file audits are unnecessary.

No additional spin/orbital difference plots are required by this plan. Report
block-resolved values only when investigating a specific symmetry or observable.
Do not convert native errors into meV or interpret offsets as Hartree terms until
units and conventions are established. Spectral/DOS claims remain separate work.

To create the comparison config on Ubuntu (does not read raw arrays or train):

```bash
conda activate physai
cd /home/frederic104/projects/dmft-self-energy
python - <<'PY'
from pathlib import Path
import yaml
src = Path('configs/experiment.yaml')
dst = Path('configs/experiment_same_iteration_1.local.yaml')
cfg = yaml.safe_load(src.read_text())
cfg['selection']['target_path'] = '/dmft_output/Sigma-1'
cfg['prepared'] = 'data/processed/learning/same_iteration_1'
cfg['experiment'] = 'reports/local/experiments/same_iteration_1'
cfg['assumptions'].append('Same-index iteration 1 comparison; not a converged target.')
with dst.open('x') as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(dst)
PY
# After reviewing any per-run selector overrides:
python -m dmft_data prepare --config configs/experiment_same_iteration_1.local.yaml
python -m dmft_data summary --config configs/experiment_same_iteration_1.local.yaml
# Confirm the displayed paths before training:
python -m dmft_data train --config configs/experiment_same_iteration_1.local.yaml
```

This is an optional local experiment, not an automatic rerun of 62 files.

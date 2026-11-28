# AIM property reconstruction from predicted self-energy

This workflow takes already predicted Matsubara Sigma and reconstructs impurity
G and a per-spin occupation without needing ED reference values. It does not
train or load a model or claim prediction accuracy on the supplied states.
Use the existing frozen evaluation and [assessment](surrogate_property_assessment.md)
for the previously measured PCA/grid/causal performance. Choosing a model,
preparing its predictions and checking the domain of applicability are separate
responsibilities; this command does not automatically select a model.

## Input contract

Supply an NPZ file containing the following arrays for the **same ordered AIM
physical states**, frequency grid, units and conventions:

| Key | Shape | Meaning |
| --- | --- | --- |
| `g0` | (states, frequencies), complex | Scalar noninteracting AIM impurity propagator |
| `sigma` | same shape, complex | Predicted impurity self-energy |
| `omega` | (frequencies,), real | At least 16 consecutive positive fermionic frequencies `(2n+1)*pi/beta`, starting at n=0 |
| `beta` | scalar, real | Shared inverse temperature in the matching convention |

These arrays must be consistent and finite. The code checks shapes, frequencies,
finite values and singular Dyson results; it cannot infer whether the arrays
actually represent the same system, whether the self-energy is accurate, or
whether the state order is correct. Preserve state identifiers and model
provenance externally: output rows retain input order, with no inferred labels.
Do not feed projected material DFT G0, an unknown Weiss function, or matrix
quantities into this scalar AIM formula as a verified material calculation.

```bash
conda activate physai
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m dmft_data.ed_predict_properties \
  --input reports/local/my_aim_predictions.npz \
  --output reports/local/my_aim_properties
```

The output directory must not already exist. `properties.npz` contains `green`,
`occupancy_raw`, `occupancy_tail`, `omega`, and `beta`. `summary.json` records
input/output hashes, code hashes, package versions, counts of out-of-range tail
occupations and positive-imaginary G entries, plus scope and limitations. Counts
are diagnostics, not proof of causality or correctness. No source-root path is
recorded. The input hash identifies the input read; it does not authenticate its
origin or verify G0/Sigma pairing.

The occupation is `1/2 + (2/beta) sum Re G(iw)` per spin, with an optional
reported fitted high-frequency tail correction. Neither version is clipped.
The tail fit can be unreliable for limited grids or unphysical predictions;
only comparison against an independent reference can quantify error. Real-axis
spectra require a validated real-frequency representation and a matched
reference; lattice DOS additionally needs k-resolved inputs and conventions.

## Reproduce a known case from saved results

This example constructs the input from the already evaluated energy-axis grid+G
seed 31. It checks row parameters and mesh before combining files; it does not
run a model, regenerate ED or create new independent test evidence. Run from the
repository root, after verifying these are the intended trusted local files:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path('reports/local/final_reserved_v1/energy')
with np.load(root / 'dataset.npz', allow_pickle=False) as d, \
     np.load(root / 'predictions_seed_31.npz', allow_pickle=False) as p:
    np.testing.assert_array_equal(d['parameters'], p['parameters'])
    np.testing.assert_array_equal(d['omega'], p['omega'])
    assert np.all(d['parameters'][:, 7] == d['parameters'][0, 7])
    target = Path('reports/local/my_aim_predictions.npz')
    with target.open('xb') as f:
        np.savez_compressed(f, g0=d['g0'],
                            sigma=p['grid_latent_8_refine_sigma_g_sigma'],
                            omega=d['omega'], beta=d['parameters'][0, 7])
PY
```

Run the command above, then optionally compare the saved output with the existing
`grid_latent_8_refine_sigma_g_green` and `..._occupancy_tail` arrays in
`predictions_seed_31.npz`. For the copies checked during implementation, max
absolute differences were 0 for G and 5.56e-17 for n across 40 states. This
verifies reuse of the saved computation, not independent model validation.

## Further work before material properties

The next material step is reference-first: establish the stored quantity meaning,
stage, basis, mesh, chemical-potential and double-counting conventions, then
compare a reference reconstruction against a compatible reference G/occupation.
Convergence is a known limitation of the public data; exploratory comparisons
against archived arrays remain possible with that limitation clearly stated.
Material property prediction needs an appropriate matrix/k-resolved workflow and
reference. Do not interpret this AIM scalar helper as that workflow.

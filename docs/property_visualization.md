# Visualizing existing self-energy and impurity-spectrum results

These views use existing repository readers, trained baselines and saved ED
outputs. They do not infer DMFT convergence or implement material DOS or bands.
All real-material physical interpretations remain subject to the documented
[source contract](upstream_data_contract.md). Do not use this workflow to reopen
the closed AIM model search or treat previously inspected states as a new test.

## Exploratory single-run material self-energy

Use `material_prediction_view` with a trusted local experiment, its original
prepared files, and a matching raw HDF5 file. Joblib uses Python pickle and can
execute code: **only load checkpoints you trust**. The command verifies the
experiment/prepared hashes, recorded shapes and blocks, raw file hash, and split;
it then predicts from that single prepared run without fitting. It uses the
existing complex encoder, trained preprocessing, and `prediction_figure`.

For the existing Ubuntu `exploratory_v1` experiment, from the repository root:

```bash
conda activate physai
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python -m dmft_data.material_prediction_view \
  --prepared data/processed/learning/exploratory_v1 \
  --experiment reports/local/experiments/exploratory_v1 \
  --source SVO/svo_beta40_nwarmup50000.h5 \
  --raw-file data/raw/hdf5/SVO/svo_beta40_nwarmup50000.h5 \
  --seed 0 --output reports/local/svo_single_run_view
```

The relative `--source` must match the prepared manifest exactly. `--raw-file`
can point to a local copy of the same HDF5; only its hash is read here. If a
file differs from the original experiment's source, the command stops. The
output directory must not already exist. Check the local experiment layout before
running: this command does not reconstruct missing prepared data or checkpoints.

`sigma_comparison.png` and its block-resolved companion show real and imaginary
parts of saved Sigma versus the model prediction on positive Matsubara points;
solid is stored reference and dashed is prediction. Separate PDF/caption files
and `summary.json` record details and hashes. No frequency interpolation, unit
conversion, spin/orbital averaging or alignment is performed. The model hash
identifies the checkpoint read now; the original experiment did not record a
creation-time model hash. The stored-array meanings and units remain unverified.

The original experiment groups replicas by candidate material and beta and trains
on group means. SVO belongs to a **training group**: this command applies the
model to a *single run*, not its group-averaged training feature. The figure and
summary label the original split. Its single-run residual is descriptive and is
neither independent generalization nor a re-evaluation of the group-level test.
Warmup labels, spin/orbital channels and model seeds are not independent states.
This view applies the Track A model; AIM models cannot be substituted for it.

## Saved AIM impurity spectral curves

This draws existing predictions and the matching broadened ED reference for one
state, without loading models, recomputing ED or performing analytic continuation:

```bash
python -m dmft_data.ed_spectrum_view \
  --input reports/local/final_spectra_v1/energy/spectra_eta_0.1.npz \
  --state 0 --output reports/local/aim_spectrum_view
```

`impurity_spectrum.png` displays all saved seed predictions and the ED reference
on the exact same grid and eta. The summary preserves input hash, state index,
parameters and eta. One state is an illustration, not an aggregate result; see
the [assessment](surrogate_property_assessment.md) for all-state metrics.
Only models with a valid real-frequency representation were saved. Absence of
PCA/grid curves is not zero error. This is a spin-averaged per-spin AIM impurity
spectral function, **not a lattice DOS**. Energy is relative to mu in native AIM
units; broadening/window affect peaks and integrated weight.

## DFT+DMFT quantities still requiring evidence

For lattice DOS or momentum-resolved spectral intensity, verify basis, spin,
orbital rotations, projectors, hopping, energy zero, chemical potential, double
counting and self-energy conventions for the actual stored run. In the usual
orthogonal Bloch representation, embed `Sigma - V_DC` with projectors, invert
at each k and then perform the k-weighted trace; averaging before inversion is
not equivalent. A band-path spectral map also needs the appropriate k-path
Hamiltonian/projectors and a validated real-frequency self-energy. A Matsubara
curve alone does not supply those. The nominated SVO file contains candidate
payloads, but the k-path and semantic alignment are not verified. Follow the
[DFTTools formulation](https://triqs.github.io/dft_tools/latest/basicnotions/dft_dmft.html)
and first compare reference-Sigma reconstructions to a compatible material
reference before interpreting model-based DOS or bands.

# Research-readiness audit

`scripts/10_audit_research_readiness.py` creates four JSON reports from local
raw HDF5 paths and optional, independently curated metadata:

- `physical_state_manifest.json`: immutable run provenance and the storage
  diagnostic already observed for each source file.
- `target_registry.json`: fail-closed selection of primary targets. A target is
  eligible only when an external registry supplies `status:
  "verified_converged"`, a non-empty `evidence_source`, and an integer
  `selected_iteration`.
- `replica_audit.json`: groups only runs whose `physical_state_id` was supplied
  explicitly. It does not infer independence from filenames, warmup, spin,
  orbitals, or iterations.
- `blocked_splits.json`: leave-one-material-out folds from verified targets and
  explicitly assigned physical states only.

Example on the Ubuntu machine:

```bash
conda activate physai
cd /home/frederic104/projects/dmft-self-energy

python scripts/10_audit_research_readiness.py \
  --source data/raw/hdf5 \
  --diagnostics reports/local/run_diagnostics_check/run_diagnostics.json \
  --metadata data/metadata/physical_states.json \
  --convergence-registry data/metadata/convergence_registry.json \
  --output reports/local/research_audit
```

Both metadata arguments are optional. Without a convergence registry, the
target registry deliberately contains zero primary targets and no ML-ready
split is produced. Do not add raw HDF5, generated reports, solver logs, or
private paths to Git.

Minimal registry shape:

```json
{
  "runs": {
    "BVO/BVO_beta40_nwarmup100000.h5": {
      "status": "verified_converged",
      "evidence_source": "solver log path and criterion reviewed by researcher",
      "selected_iteration": 15,
      "selected_quantity_paths": {
        "G_iw": "/dmft_output/G_iw",
        "Sigma_iw": "/dmft_output/Sigma_iw"
      }
    }
  }
}
```

The registry is evidence supplied by a researcher; this repository records it
but does not independently establish convergence.

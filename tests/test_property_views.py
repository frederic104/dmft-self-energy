"""Check saved-spectrum and prepared-run visualization boundaries."""

import json

import numpy as np
import pytest

from dmft_data.ed_spectrum_view import visualize as spectrum_view
from dmft_data.learning.data import prepare
from dmft_data.learning.experiment import run_experiment
from dmft_data.material_prediction_view import visualize as material_view
from test_learning import setup_experiment


def test_spectrum_view_uses_one_state_and_saved_seeds(tmp_path):
    x = np.linspace(-4, 4, 101)
    ref = np.stack([1/(np.pi*(1+x*x)), 1/(np.pi*(1+(x-1)**2))])
    src = tmp_path / "spectrum.npz"
    np.savez(src, energy=x, eta=.1, reference_spectrum=ref, parameters=np.ones((2, 8)),
             seed_31_causal_spectrum=ref+.02, seed_32_causal_spectrum=ref+.01)
    out = tmp_path / "out"
    result = spectrum_view(src, out, 1)
    assert result["state_index"] == 1 and len(result["predictions"]) == 2
    assert (out / "impurity_spectrum.png").exists()
    with pytest.raises(ValueError, match="state"):
        spectrum_view(src, tmp_path / "bad", 2)
    with pytest.raises(FileExistsError):
        spectrum_view(src, out, 0)


def test_prepared_run_view_records_split_and_checks_source(tmp_path):
    cfg = setup_experiment(tmp_path)
    cfg["models"] = [{"name": "mean"}]
    cfg["seeds"] = [0]
    prepare(cfg, tmp_path)
    run_experiment(cfg, tmp_path)
    source = "SVO/run0.h5"
    raw = tmp_path / "raw" / source
    output = tmp_path / "view"
    result = material_view(tmp_path / "prepared", tmp_path / "experiment", source, raw, output)
    assert result["split"] == "train"
    assert result["input_path"] == "/dmft_output/G0-1"
    assert result["target_path"] == "/dmft_output/Sigma_iw"
    assert result == json.loads((output / "summary.json").read_text())
    assert (output / "sigma_comparison.png").exists()
    assert str(tmp_path) not in json.dumps(result)
    with pytest.raises(ValueError, match="checksum"):
        material_view(tmp_path / "prepared", tmp_path / "experiment", source,
                      tmp_path / "raw/SNO/run0.h5", tmp_path / "wrong")

"""Independent analytic limits and gauge covariance for lattice postprocessing."""

import json

import numpy as np
import pytest
import yaml

from dmft_data.lattice import bz_average, lattice_green, process, run


def conventions():
    return dict(basis="orthonormal", sigma_convention="impurity_before_dc",
                energy_axis="relative_to_mu", spin="one_explicit_block",
                energy_unit="model_units", interpretation="synthetic",
                sampling="bz", domain="real_frequency")


def fixture():
    z = np.linspace(-4, 4, 401) + .1j
    return dict(hopping=np.array([[[-1.]], [[1.]]]), projectors=np.ones((2, 1, 1)),
                sigma=np.zeros((len(z), 1, 1), complex), dc=np.zeros((1, 1)),
                z=z, mu=np.array(.2), weights=np.array([.3, .7]))


def test_noninteracting_dos_is_weighted_lorentzian_and_no_spin_factor():
    a = fixture()
    r = process(a, conventions())
    e = a["z"].real
    expected = sum(w * .1 / (np.pi * ((e + .2 - level)**2 + .1**2))
                   for w, level in zip(a["weights"], [-1., 1.]))
    np.testing.assert_allclose(r["dos"], expected, rtol=1e-13)
    np.testing.assert_allclose(r["projected_dos"], expected)
    # Matching Sigma and DC must cancel once, without a second subtraction.
    a["sigma"][:] = .8
    a["dc"][:] = .8
    np.testing.assert_allclose(process(a, conventions())["dos"], expected)


def test_invert_before_average_and_energy_zero_invariance():
    a = fixture()
    a["sigma"][:] = .4
    result = process(a, conventions())
    g0 = 1/(a["z"][None, :] + .2 - a["hopping"][:, 0, 0, None])
    wrong = 1/(1/bz_average(g0, a["weights"]) - .4)
    assert np.max(abs(wrong - result["green_local"][:, 0, 0])) > .1
    a["hopping"] += 2.3
    a["mu"] += 2.3
    np.testing.assert_allclose(process(a, conventions())["green_k"], result["green_k"])


def test_rational_self_energy_matches_auxiliary_hamiltonian_resolvent():
    # Independent Schur-complement reference with complex orbital mixing.
    h = np.array([[[.1, .2j], [-.2j, -.4]]])
    p = np.array([[[1., 0.]]])
    z = np.linspace(-3, 3, 31) + .07j
    coupling, level, static, dc = .6, 1.2, .5, .1
    sigma = (static + coupling**2/(z-level))[:, None, None]
    result = lattice_green(h, p, sigma, z, 0., np.array([[dc]]))
    augmented = np.zeros((3, 3), complex)
    augmented[:2, :2] = h[0]
    augmented[0, 0] += static-dc
    augmented[2, 2] = level
    augmented[0, 2] = augmented[2, 0] = coupling
    exact = np.array([np.linalg.inv(w*np.eye(3)-augmented)[:2, :2] for w in z])
    np.testing.assert_allclose(result["green_k"][0], exact, atol=1e-13)
    spectral = -(exact-exact.swapaxes(-1, -2).conj())/(2j*np.pi)
    assert np.min(np.linalg.eigvalsh(spectral)) > 0


def test_complex_unitary_gauge_covariance_and_high_frequency_moment():
    rng = np.random.default_rng(8)
    def unitary(n):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))
        return q
    u, v = unitary(2), unitary(3)
    h = np.array([np.diag([-.8, .3, 1.2])], complex)
    p = np.array([[[1, 0, 0], [0, 1, 0]]], complex)
    z = np.array([.4+.1j, 1e7j])
    s = np.array([np.diag([.3, .7]) + np.diag([.2, .5])/(w-1.5) for w in z])
    dc = np.diag([.1, .2])
    a = lattice_green(h, p, s, z, .15, dc)
    b = lattice_green(v @ h @ v.conj().T, u @ p @ v.conj().T,
                      u @ s @ u.conj().T, z, .15, u @ dc @ u.conj().T)
    np.testing.assert_allclose(b["green_k"], v @ a["green_k"] @ v.conj().T, atol=1e-13)
    np.testing.assert_allclose(b["projected_green_k"], u @ a["projected_green_k"] @ u.conj().T, atol=1e-13)
    np.testing.assert_allclose(z[-1]*a["green_k"][0, -1], np.eye(3), atol=2e-7)
    # Incorrectly rotating P alone is detectable for a nondegenerate Sigma.
    bad = lattice_green(h, u @ p, s, z, .15, dc)
    assert np.max(abs(bad["green_k"] - a["green_k"])) > .1


def test_domains_path_not_dos_and_invalid_input_rejected():
    a = fixture()
    c = conventions()
    c["sampling"] = "path"
    a["k_distance"] = np.array([0., 1.])
    with pytest.raises(ValueError, match="not a BZ"):
        process(a, c)
    del a["weights"]
    r = process(a, c)
    assert "spectral_k" in r and "dos" not in r and "green_local" not in r
    c["domain"] = "matsubara"
    a["z"] = 1j * np.arange(1, 402)
    assert "spectral_k" not in process(a, c)
    a["z"][0] = -1j
    with pytest.raises(ValueError, match="upper-half-plane"):
        process(a, c)
    with pytest.raises(ValueError, match="sum to one"):
        bz_average(np.ones((2, 3)), [.1, .1])
    a = fixture()
    c = conventions()
    c["sigma_convention"] = "already_dc_subtracted"
    with pytest.raises(ValueError, match="contract"):
        process(a, c)
    c = conventions()
    a["dc"] = np.array([[1j]])
    with pytest.raises(ValueError, match="Hermitian"):
        process(a, c)


def test_run_records_hashes_without_root_paths(tmp_path):
    source, cfg, out = tmp_path/"input.npz", tmp_path/"config.yaml", tmp_path/"result"
    np.savez(source, **fixture())
    cfg.write_text(yaml.safe_dump(conventions()))
    s = run(source, cfg, out)
    assert s == json.loads((out/"summary.json").read_text())
    assert s["material_validation"] == "not_assessed"
    assert s["negative_spectral_trace_count"] == 0
    assert str(tmp_path) not in json.dumps(s)
    with pytest.raises(FileExistsError):
        run(source, cfg, out)

"""Finite-temperature exact diagonalization reference for a small AIM.

The model has one spinful impurity and two spin-degenerate bath sites.  It is a
controlled impurity reference only: its broadened impurity spectrum is not a
lattice density of states and it does not implement a DMFT self-consistency.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AIMParameters:
    """Parameters of ``H`` (not of ``H - mu*N``), in one common energy unit."""

    epsilon_d: float
    interaction_u: float
    bath_energies: tuple[float, float]
    hybridizations: tuple[float, float]
    chemical_potential: float = 0.0
    beta: float = 20.0

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if len(self.bath_energies) != 2 or len(self.hybridizations) != 2:
            raise ValueError("this reference has exactly two bath sites")


@dataclass(frozen=True)
class EDResult:
    """Thermal eigensystem plus impurity operators in the eigenbasis."""

    parameters: AIMParameters
    energies: np.ndarray
    weights: np.ndarray
    impurity_annihilation: tuple[np.ndarray, np.ndarray]


def _annihilation(orbital: int, n_orbitals: int = 6) -> np.ndarray:
    """Return c_orbital in ascending-orbital fermionic Fock convention."""
    dimension = 1 << n_orbitals
    operator = np.zeros((dimension, dimension), dtype=complex)
    mask = (1 << orbital) - 1
    for state in range(dimension):
        if state & (1 << orbital):
            destination = state ^ (1 << orbital)
            sign = -1 if ((state & mask).bit_count() % 2) else 1
            operator[destination, state] = sign
    return operator


def solve(params: AIMParameters) -> EDResult:
    """Diagonalize K = H - mu*N in the full 64-state Fock space.

    Orbital order is ``(d_up, d_down, b0_up, b0_down, b1_up, b1_down)`` and
    ``H = eps_d n_d + U n_d_up n_d_down + sum(eps_l n_l +
    V_l(d^dag c_l + c_l^dag d))``.  The grand-canonical eigenvalues returned
    here include the supplied chemical potential.
    """
    annihilation = tuple(_annihilation(index) for index in range(6))
    number = tuple(op.conj().T @ op for op in annihilation)
    hamiltonian = (params.epsilon_d - params.chemical_potential) * (
        number[0] + number[1]
    )
    hamiltonian = hamiltonian + params.interaction_u * (number[0] @ number[1])
    for bath, (energy, coupling) in enumerate(
        zip(params.bath_energies, params.hybridizations, strict=True)
    ):
        for spin in range(2):
            bath_orbital = 2 + 2 * bath + spin
            impurity_orbital = spin
            hamiltonian = hamiltonian + (energy - params.chemical_potential) * number[
                bath_orbital
            ]
            hamiltonian = hamiltonian + coupling * (
                annihilation[impurity_orbital].conj().T @ annihilation[bath_orbital]
                + annihilation[bath_orbital].conj().T @ annihilation[impurity_orbital]
            )
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    shifted = energies - energies.min()
    weights = np.exp(-params.beta * shifted)
    weights /= weights.sum()
    d_eigenbasis = tuple(
        eigenvectors.conj().T @ annihilation[spin] @ eigenvectors for spin in range(2)
    )
    return EDResult(params, energies, weights, d_eigenbasis)


def matsubara_frequencies(beta: float, count: int) -> np.ndarray:
    """Return the first ``count`` positive fermionic Matsubara frequencies."""
    if beta <= 0 or count < 1:
        raise ValueError("beta must be positive and count must be at least one")
    return (2 * np.arange(count) + 1) * np.pi / beta


def _lehmann_green(result: EDResult, z: np.ndarray, spin: int) -> np.ndarray:
    energies = result.energies
    numerator = result.weights[:, None] + result.weights[None, :]
    # Rows m and columns n represent <m|d|n>; denominator z + E_m - E_n.
    transition_weight = np.abs(result.impurity_annihilation[spin]) ** 2 * numerator
    differences = energies[:, None] - energies[None, :]
    # Limit temporary memory for dense real-frequency grids.
    chunks = []
    for start in range(0, len(z), 128):
        points = z[start : start + 128, None, None]
        chunks.append(np.sum(transition_weight / (points + differences), axis=(1, 2)))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=complex)


def green_iw(result: EDResult, omega: np.ndarray, spin: int | None = None) -> np.ndarray:
    """Return Lehmann impurity G(i omega), spin averaged by default."""
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1:
        raise ValueError("omega must be a one-dimensional frequency array")
    return green_z(result, 1j * omega, spin)


def green_z(result: EDResult, z: np.ndarray, spin: int | None = None) -> np.ndarray:
    """Return retarded/complex-frequency impurity Green's function by Lehmann ED."""
    z = np.asarray(z, dtype=complex)
    if z.ndim != 1:
        raise ValueError("z must be a one-dimensional complex-frequency array")
    spins = range(2) if spin is None else (spin,)
    if any(value not in (0, 1) for value in spins):
        raise ValueError("spin must be 0, 1, or None")
    values = sum(_lehmann_green(result, z, value) for value in spins)
    return values / len(tuple(spins))


def noninteracting_green_iw(params: AIMParameters, omega: np.ndarray) -> np.ndarray:
    """Return analytic impurity G0 for the same finite bath and convention."""
    z = 1j * np.asarray(omega, dtype=float) + params.chemical_potential
    hybridization = sum(
        coupling**2 / (z - energy)
        for energy, coupling in zip(
            params.bath_energies, params.hybridizations, strict=True
        )
    )
    return 1.0 / (z - params.epsilon_d - hybridization)


def self_energy_iw(result: EDResult, omega: np.ndarray) -> np.ndarray:
    """Return Sigma = G0^-1 - G^-1, with no double-counting correction."""
    green = green_iw(result, omega)
    g0 = noninteracting_green_iw(result.parameters, omega)
    return 1.0 / g0 - 1.0 / green


def spectral_function(result: EDResult, energy: np.ndarray, broadening: float) -> np.ndarray:
    """Return A_imp(omega) = -Im G^R(omega+i eta)/pi for eta > 0."""
    if broadening <= 0:
        raise ValueError("broadening must be positive")
    values = green_z(result, np.asarray(energy, dtype=float) + 1j * broadening)
    return -values.imag / np.pi


def generate_small_dataset(
    output: Path,
    *,
    samples: int = 8,
    seed: int = 0,
    beta: float = 20.0,
    frequency_count: int = 64,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate a deterministic small AIM reference set in a compressed NPZ file."""
    if samples < 1:
        raise ValueError("samples must be at least one")
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("output must have a .npz extension")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    omega = matsubara_frequencies(beta, frequency_count)
    parameters, greens, g0s, sigmas = [], [], [], []
    for _ in range(samples):
        params = AIMParameters(
            epsilon_d=float(rng.uniform(-1.0, 0.2)),
            interaction_u=float(rng.uniform(0.5, 4.0)),
            bath_energies=tuple(float(x) for x in rng.uniform(-2.0, 2.0, 2)),
            hybridizations=tuple(float(x) for x in rng.uniform(0.15, 1.0, 2)),
            beta=beta,
        )
        result = solve(params)
        parameters.append(
            [
                params.epsilon_d,
                params.interaction_u,
                *params.bath_energies,
                *params.hybridizations,
                params.chemical_potential,
                params.beta,
            ]
        )
        green = green_iw(result, omega)
        g0 = noninteracting_green_iw(params, omega)
        greens.append(green)
        g0s.append(g0)
        sigmas.append(1.0 / g0 - 1.0 / green)
    metadata = {
        "model": "finite_bath_AIM_1_impurity_2_bath_spinful",
        "samples": samples,
        "seed": seed,
        "beta": beta,
        "frequency_count": frequency_count,
        "numpy_version": np.__version__,
        "convention": "K=H-mu*N; Sigma=G0^-1-G_imp^-1; spin_averaged",
        "limitation": "Impurity ED reference; not lattice DMFT or lattice DOS.",
    }
    np.savez_compressed(
        output,
        omega=omega,
        parameters=np.asarray(parameters),
        green=np.asarray(greens),
        g0=np.asarray(g0s),
        sigma=np.asarray(sigmas),
        metadata=np.asarray(json.dumps(metadata)),
        parameter_columns=np.asarray(
            [
                "epsilon_d",
                "interaction_u",
                "epsilon_b0",
                "epsilon_b1",
                "V_b0",
                "V_b1",
                "chemical_potential",
                "beta",
            ]
        ),
    )
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--frequency-count", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = generate_small_dataset(
        args.output,
        samples=args.samples,
        seed=args.seed,
        beta=args.beta,
        frequency_count=args.frequency_count,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

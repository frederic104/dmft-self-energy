"""Impurity properties from predicted self-energy; no analytic continuation."""

from __future__ import annotations

import numpy as np

from .ed import matsubara_frequencies


def dyson_green(g0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Reconstruct scalar impurity G on the supplied common frequency mesh."""
    if g0.shape != sigma.shape or not np.isfinite(g0).all() or not np.isfinite(sigma).all():
        raise ValueError("Dyson inputs must be finite and have the same shape")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        green = 1 / (1 / g0 - sigma)
    if np.any(g0 == 0) or not np.isfinite(green).all():
        raise ValueError("Singular or non-finite Dyson reconstruction")
    return green


def occupancy_estimates(green: np.ndarray, omega: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-spin n = 1/2 + (2/beta) sum_{positive n} Re G(iw_n).

    Return the finite sum and a fitted-tail estimate. Fit Re G=-m1/w^2+m3/w^4
    on the last 8-16 points and analytically sum those two omitted terms.
    This asymptotic approximation is evaluated against independent ED occupation;
    it is not guaranteed for short meshes or arbitrary nonphysical predictions.
    No reference occupancy or reference moments enter the estimate.
    """
    if (green.ndim != 2 or omega.ndim != 1 or green.shape[1] != len(omega)
            or len(omega) < 16 or not np.isfinite(green).all() or not np.isfinite(beta)
            or beta <= 0 or not np.allclose(omega, matsubara_frequencies(beta, len(omega)))):
        raise ValueError("Occupancy needs finite G and at least 16 consecutive positive Matsubara points")
    raw = .5 + 2 / beta * green.real.sum(axis=1)
    count = min(16, len(omega) // 2)
    tail = omega[-count:]
    scale = tail[0]
    design = np.column_stack((-(scale / tail)**2, (scale / tail)**4))
    coefficients = np.linalg.lstsq(design, green[:, -count:].real.T, rcond=None)[0]
    m1, m3 = coefficients[0] * scale**2, coefficients[1] * scale**4
    omitted2 = max(0., beta**2 / 8 - np.sum(1 / omega**2))
    omitted4 = max(0., beta**4 / 96 - np.sum(1 / omega**4))
    corrected = raw + 2 / beta * (-m1 * omitted2 + m3 * omitted4)
    if not np.isfinite(corrected).all():
        raise ValueError("Non-finite occupancy estimate")
    return raw, corrected


def compare_properties(g0: np.ndarray, sigma: np.ndarray, reference_green: np.ndarray,
                       exact_occupancy: np.ndarray, omega: np.ndarray, beta: float) -> tuple[dict, dict]:
    """Evaluate model propagation and reference finite-mesh error separately."""
    if (reference_green.shape != sigma.shape or exact_occupancy.shape != (len(sigma),)
            or not np.isfinite(exact_occupancy).all()):
        raise ValueError("Invalid property reference shape or occupancy")
    predicted = dyson_green(g0, sigma)
    raw, estimated = occupancy_estimates(predicted, omega, beta)
    ref_raw, ref_estimated = occupancy_estimates(reference_green, omega, beta)

    def rmse(value):
        return float(np.sqrt(np.mean(np.abs(value)**2)))

    metrics = {
        "status": "ok",
        "green_complex_rmse": rmse(predicted - reference_green),
        "green_first_8_complex_rmse": rmse(predicted[:, :8] - reference_green[:, :8]),
        "green_positive_imaginary_fraction": float(np.mean(predicted.imag > 1e-10)),
        "occupancy_raw_vs_exact_rmse": rmse(raw - exact_occupancy),
        "occupancy_tail_vs_exact_rmse": rmse(estimated - exact_occupancy),
        "occupancy_tail_model_vs_reference_method_rmse": rmse(estimated - ref_estimated),
        "reference_raw_vs_exact_rmse": rmse(ref_raw - exact_occupancy),
        "reference_tail_vs_exact_rmse": rmse(ref_estimated - exact_occupancy),
        "occupancy_tail_out_of_bounds_fraction": float(np.mean((estimated < 0) | (estimated > 1))),
    }
    return metrics, {"green": predicted, "occupancy_raw": raw, "occupancy_tail": estimated}

"""Fixed-real-pole scalar self-energy decoder for a controlled AIM comparison."""

from copy import deepcopy

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import expit

from .models import Baseline, encode


class PoleBaseline(Baseline):
    """Real constant plus positive residues on a prespecified real pole grid."""

    def transform_output(self, raw):
        coefficients = np.column_stack((raw[:, 0], np.logaddexp(0, raw[:, 1:])))
        return self.y_scaler.transform(coefficients @ self.basis.T)

    def output_gradient(self, raw, gradient):
        result = (gradient / self.y_scaler.scale_) @ self.basis
        result[:, 1:] *= expit(raw[:, 1:])
        return result

    def predict(self, x):
        raw = self.model.predict(self.x_scaler.transform(x))
        return self.y_scaler.inverse_transform(self.transform_output(raw))


def decoder_starts(template, sigma, omega, *, seed, pole_count=16, pole_bound=4.):
    """Reset both networks; reuse only architecture and training-fitted scalers.

    Hidden architecture and initial hidden weights are identical. Output heads
    differ in dimension/parameterization; counts are recorded, not equated.
    The grid head starts near the training mean; the pole head near its bounded
    least-squares representation. Each loss treatment later clones these starts.
    """
    if template.specification.get("name") != "latent_mlp" or template.actual_dimension != 8:
        raise ValueError("Decoder comparison requires the saved latent_mlp_8 template")
    if pole_count < 2 or not np.isfinite(pole_bound) or pole_bound <= 0:
        raise ValueError("Need at least two poles and a positive finite bound")
    omega = np.asarray(omega)
    if omega.ndim != 1 or not np.isfinite(omega).all() or np.any(omega <= 0):
        raise ValueError("Expected positive finite frequencies")
    grid = deepcopy(template)
    rng = np.random.default_rng(seed)
    for i, w in enumerate(grid.model.coefs_):
        limit = np.sqrt(6 / sum(w.shape))
        grid.model.coefs_[i] = rng.uniform(-limit, limit, w.shape)
        grid.model.intercepts_[i] = np.zeros(w.shape[1])
    grid.model.coefs_[-1] *= .01
    poles = np.linspace(-pole_bound, pole_bound, pole_count)
    complex_basis = np.column_stack((np.ones(len(omega)), 1 / (1j * omega[:, None] - poles)))
    basis = np.concatenate((complex_basis.real, complex_basis.imag), axis=0)
    design = basis / template.y_scaler.scale_[:, None]
    lower = np.r_[-np.inf, np.zeros(pole_count)]
    fits = [lsq_linear(design, y / template.y_scaler.scale_, bounds=(lower, np.inf), method="bvls")
            for y in encode(sigma)]
    mean_fit = lsq_linear(design, encode(sigma).mean(axis=0) / template.y_scaler.scale_,
                          bounds=(lower, np.inf), method="bvls")
    pole = PoleBaseline(deepcopy(template.specification), seed)
    pole.__dict__.update(deepcopy(grid.__dict__))
    pole.poles, pole.basis = poles, basis
    width = grid.model.coefs_[-1].shape[0]
    head_rng = np.random.default_rng(seed + 1)
    pole.model.coefs_[-1] = head_rng.normal(scale=.001, size=(width, pole_count + 1))
    # Positive initialization floor only; predictions and gradients are never clipped.
    residues = np.maximum(mean_fit.x[1:], 1e-8)
    pole.model.intercepts_[-1] = np.r_[mean_fit.x[0], residues + np.log(-np.expm1(-residues))]
    pole.model.n_outputs_ = pole_count + 1
    reconstruction = np.stack([f.x for f in fits]) @ basis.T
    error = reconstruction - encode(sigma)
    real, imag = np.split(error, 2, axis=1)
    info = {"pole_count": pole_count, "poles": poles.tolist(), "latent_dimension": 8,
            "initialization": "reset shared hidden weights; mean-based output heads; no pretrained weights",
            "training_representation_rmse": float(np.sqrt(np.mean(real**2 + imag**2))),
            "representation_fit_success": all(f.success for f in fits) and bool(mean_fit.success),
            "representation_fit_failures": [{"training_row": i, "message": str(f.message)}
                                            for i, f in enumerate(fits) if not f.success],
            "mean_fit_status": {"success": bool(mean_fit.success), "message": str(mean_fit.message)},
            "parameter_counts": {name: sum(a.size for a in m.model.coefs_ + m.model.intercepts_)
                                 for name, m in (("grid", grid), ("causal", pole))}}
    return {"grid_latent_8": grid, "causal_latent_8": pole}, info

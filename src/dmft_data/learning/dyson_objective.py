"""Matched CPU refinement of existing MLPs with an analytic Dyson gradient."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

from ..ed_properties import dyson_green
from .models import decode, encode


def output_loss(prediction, target, scaler, g0, green, green_scale2, weight):
    """Loss and derivative w.r.t. standardized real/imaginary Sigma outputs.

    L_sigma = mean((standardized Sigma prediction - target)**2).
    L_G = mean(|G_prediction-G_reference|**2) / training mean(|G_reference|**2).
    For complex e=G_prediction-G_reference, dG/dSigma=G_prediction**2;
    real/imaginary Sigma derivatives are 2 Re(conj(e)*G**2) and
    -2 Im(conj(e)*G**2), respectively. No clipping or reference tail is used.
    """
    residual = prediction - target
    sigma_loss = float(np.mean(residual**2))
    gradient = 2 * residual / residual.size
    green_loss = 0.
    if weight:
        sigma = decode(scaler.inverse_transform(prediction), g0.shape[1:])
        reconstructed = dyson_green(g0, sigma)
        error = reconstructed - green
        green_loss = float(np.mean(np.abs(error)**2) / green_scale2)
        factor = 2 * error.conj() * reconstructed**2 / (error.size * green_scale2)
        gradient += weight * np.concatenate((factor.real, -factor.imag), axis=1) * scaler.scale_
    return sigma_loss + weight * green_loss, gradient, (sigma_loss, green_loss)


class NetworkObjective:
    """Full-batch loss over training arrays only; public sklearn fitted weights."""

    def __init__(self, baseline, x, sigma, g0, green, weight, alpha):
        self.scaler = baseline.y_scaler
        self.baseline = baseline
        self.x = baseline.x_scaler.transform(x)
        self.target = self.scaler.transform(encode(sigma))
        self.g0, self.green = g0, green
        self.green_scale2 = float(np.mean(np.abs(green)**2))
        if not np.isfinite(self.green_scale2) or self.green_scale2 <= 0:
            raise ValueError("Training Green function must have finite nonzero norm")
        self.weight, self.alpha = weight, alpha
        arrays = baseline.model.coefs_ + baseline.model.intercepts_
        self.shapes = [a.shape for a in arrays]
        self.sizes = [a.size for a in arrays]
        self.layers = len(baseline.model.coefs_)
        self.initial = np.concatenate([a.ravel() for a in arrays])

    def unpack(self, vector):
        parts = np.split(vector, np.cumsum(self.sizes)[:-1])
        arrays = [p.reshape(shape) for p, shape in zip(parts, self.shapes)]
        return arrays[:self.layers], arrays[self.layers:]

    def evaluate(self, vector):
        weights, biases = self.unpack(vector)
        activations = [self.x]
        for i, (w, b) in enumerate(zip(weights, biases)):
            value = activations[-1] @ w + b
            activations.append(np.maximum(value, 0) if i < self.layers - 1 else value)
        prediction = (self.baseline.transform_output(activations[-1])
                      if hasattr(self.baseline, "transform_output") else activations[-1])
        loss, delta, parts = output_loss(prediction, self.target, self.scaler,
                                       self.g0, self.green, self.green_scale2, self.weight)
        if hasattr(self.baseline, "output_gradient"):
            delta = self.baseline.output_gradient(activations[-1], delta)
        penalty = self.alpha / (2 * len(self.x)) * sum(np.sum(w**2) for w in weights)
        dw, db = [None] * self.layers, [None] * self.layers
        for i in reversed(range(self.layers)):
            dw[i] = activations[i].T @ delta + self.alpha / len(self.x) * weights[i]
            db[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ weights[i].T) * (activations[i] > 0)
        gradient = np.concatenate([a.ravel() for a in dw + db])
        if not np.isfinite(loss + penalty) or not np.isfinite(gradient).all():
            raise ValueError("Non-finite objective or gradient; no clipping applied")
        return float(loss + penalty), gradient, parts


def refine_model(baseline, x, sigma, g0, green, *, weight=1., max_iter=100, alpha=1e-3):
    """Clone one saved MLP; never modify its preprocessing or original weights.

    Each treatment must start from the same baseline, including weight=0 control.
    Validation is not used for scaling, optimization, stopping or checkpointing.
    """
    if (baseline.specification["name"] not in {"mlp", "latent_mlp"}
            or baseline.model.activation != "relu" or baseline.model.out_activation_ != "identity"):
        raise ValueError("Refinement supports fitted ReLU MLP baselines only")
    if not np.isfinite(weight) or weight < 0 or max_iter < 1 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid refinement budget, weight or regularization")
    if (sigma.shape != g0.shape or green.shape != g0.shape or len(x) != len(sigma)
            or not len(x) or not all(np.isfinite(a).all() for a in (x, sigma, g0, green))):
        raise ValueError("Invalid training arrays")
    model = deepcopy(baseline)
    objective = NetworkObjective(model, x, sigma, g0, green, weight, alpha)
    initial_loss, _, _ = objective.evaluate(objective.initial)
    history = [{"iteration": 0, "loss": initial_loss}]

    def callback(vector):
        loss, _, (sigma_loss, green_loss) = objective.evaluate(vector)
        history.append({"iteration": len(history), "loss": loss,
                        "sigma_loss": sigma_loss, "green_loss_contribution": weight * green_loss})

    result = minimize(lambda v: objective.evaluate(v)[:2], objective.initial,
                      jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": max_iter, "maxfun": max_iter * 20,
                               "ftol": 1e-9, "gtol": 1e-6, "maxls": 20})
    weights, biases = objective.unpack(result.x)
    model.model.coefs_ = [a.copy() for a in weights]
    model.model.intercepts_ = [a.copy() for a in biases]
    final_loss, _, _ = objective.evaluate(result.x)
    info = {"weight": weight, "alpha": alpha, "optimizer": "L-BFGS-B",
            "ftol": 1e-9, "gtol": 1e-6, "maxls": 20,
            "max_iter": max_iter, "maxfun": max_iter * 20, "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "success": bool(result.success),
            "message": str(result.message), "initial_loss": initial_loss,
            "final_loss": final_loss, "green_scale2": objective.green_scale2,
            "sigma_scale": "existing training-only StandardScaler, per real/imag coordinate",
            "history": history}
    model.refinement = info
    return model, info

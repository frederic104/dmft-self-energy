"""CPU baselines with training-only preprocessing and persisted predictors."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def encode(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return np.concatenate([flat.real, flat.imag], axis=1)


def decode(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    real, imag = np.split(values, 2, axis=1)
    return (real + 1j * imag).reshape((len(values), *shape))


class EpsilonLinear:
    """Piecewise linear target interpolation; endpoint slopes extrapolate."""

    def fit(self, epsilon: np.ndarray, targets: np.ndarray):
        epsilon = np.asarray(epsilon, dtype=float).reshape(-1)
        if len(epsilon) < 2 or len(epsilon) != len(targets):
            raise ValueError("Interpolation requires at least two training states")
        order = np.argsort(epsilon)
        self.epsilon = epsilon[order].copy()
        self.targets = np.asarray(targets)[order].copy()
        if not np.isfinite(self.epsilon).all() or np.any(np.diff(self.epsilon) <= 0):
            raise ValueError("Training epsilon_d values must be finite and unique")
        self.model = None
        return self

    def predict(self, epsilon: np.ndarray) -> np.ndarray:
        values = np.asarray(epsilon, dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Prediction epsilon_d must be finite")
        left = np.clip(np.searchsorted(self.epsilon, values, side="right") - 1,
                       0, len(self.epsilon) - 2)
        weight = ((values - self.epsilon[left]) /
                  (self.epsilon[left + 1] - self.epsilon[left]))
        return self.targets[left] + weight[:, None] * (self.targets[left + 1] - self.targets[left])

    def coverage(self, epsilon: np.ndarray) -> dict:
        values = np.asarray(epsilon).reshape(-1)
        return {"below": int(np.sum(values < self.epsilon[0])),
                "within_inclusive": int(np.sum((values >= self.epsilon[0]) &
                                               (values <= self.epsilon[-1]))),
                "above": int(np.sum(values > self.epsilon[-1]))}


class Baseline:
    """Mean, nearest neighbor, target-PCA+Ridge, or an MLP.

    The PCA baseline compresses training self-energies, maps inputs to their
    latent coefficients with Ridge, and decodes with the same training PCA.
    """

    def __init__(self, specification: dict, seed: int):
        self.specification = specification
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x_scaler = StandardScaler().fit(x)
        self.y_scaler = StandardScaler().fit(y)
        xs = self.x_scaler.transform(x)
        ys = self.y_scaler.transform(y)
        self.mean = y.mean(axis=0)
        self.pca = None
        name = self.specification["name"]
        if name == "mean":
            self.model = None
        elif name == "nearest":
            self.model = KNeighborsRegressor(n_neighbors=1).fit(xs, ys)
        elif name == "pca_ridge":
            requested = self.specification.get("latent_dimension", 4)
            self.actual_dimension = min(requested, len(x) - 1, y.shape[1])
            if self.actual_dimension < 1:
                raise ValueError("PCA requires at least two training groups")
            self.pca = PCA(n_components=self.actual_dimension, svd_solver="full")
            latent = self.pca.fit_transform(ys)
            self.model = Ridge(alpha=self.specification.get("alpha", 1.0)).fit(
                xs, latent
            )
        elif name in {"mlp", "latent_mlp"}:
            hidden = tuple(self.specification.get("hidden_layers", [64]))
            if name == "latent_mlp":
                dimension = int(self.specification["latent_dimension"])
                if dimension < 1:
                    raise ValueError("latent_dimension must be positive")
                hidden = (64, dimension, 64)
                self.actual_dimension = dimension
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation=self.specification.get("activation", "relu"),
                alpha=self.specification.get("alpha", 1e-3),
                max_iter=self.specification.get("max_iter", 500),
                random_state=self.seed,
                early_stopping=False,
            ).fit(xs, ys)
        else:
            raise ValueError(f"Unknown model: {name}")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.repeat(self.mean[None, :], len(x), axis=0)
        result = self.model.predict(self.x_scaler.transform(x))
        if result.ndim == 1:
            result = result[:, None]
        if self.pca is not None:
            result = self.pca.inverse_transform(result)
        return self.y_scaler.inverse_transform(result)

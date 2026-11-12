"""Фабрика моделей: (pipeline, сетка гиперпараметров для GridSearchCV)."""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.decomposition import PCA

from .preprocess import get_preprocessor


class PLS1D(PLSRegression):
    """PLSRegression, который возвращает 1D-предсказание (sklearn отдаёт (n,1))."""
    def predict(self, X):
        return np.ravel(super().predict(X))


def _pls_max(n_features, n_train_hint=30):
    return int(min(15, n_features, max(2, n_train_hint - 2)))


def build(model: str, prep: str, n_features: int):
    """Возвращает (pipeline, param_grid). Предобработка - первый шаг пайплайна."""
    pre = ("prep", get_preprocessor(prep))
    if model == "pls":
        pipe = Pipeline([pre, ("m", PLS1D(scale=False))])
        grid = {"m__n_components": list(range(1, _pls_max(n_features) + 1))}
    elif model == "ridge":
        pipe = Pipeline([pre, ("sc", StandardScaler()), ("m", Ridge())])
        grid = {"m__alpha": np.logspace(-3, 4, 8)}
    elif model == "pcr_ridge":
        pipe = Pipeline([pre, ("sc", StandardScaler()), ("pca", PCA()), ("m", Ridge(alpha=1e-6))])
        grid = {"pca__n_components": list(range(1, _pls_max(n_features) + 1))}
    elif model == "svr":
        pipe = Pipeline([pre, ("sc", StandardScaler()), ("pca", PCA(n_components=min(20, n_features))), ("m", SVR(kernel="rbf"))])
        grid = {"m__C": [1, 10, 100, 1000], "m__gamma": ["scale", 1e-3, 1e-2], "m__epsilon": [0.01, 0.1]}
    elif model == "gpr":
        k = C(1.0) * RBF(length_scale=10.0) + WhiteKernel(1e-2)
        pipe = Pipeline([pre, ("sc", StandardScaler()), ("pca", PCA(n_components=min(10, n_features))),
                         ("m", GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=1, random_state=0))])
        grid = {}
    elif model == "gbm":
        pipe = Pipeline([pre, ("m", HistGradientBoostingRegressor(random_state=0))])
        grid = {"m__max_depth": [3, None], "m__learning_rate": [0.05, 0.1], "m__max_iter": [200, 400]}
    else:
        raise KeyError(model)
    return pipe, grid


MODELS = ["pls", "ridge", "pcr_ridge", "svr", "gpr", "gbm"]

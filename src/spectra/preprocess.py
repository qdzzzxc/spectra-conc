"""Предобработка спектров как sklearn-трансформеры (работают построчно, без утечки между train/test,
кроме MSC, у которого эталон = средний спектр обучающей выборки)."""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, FunctionTransformer


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: вычесть среднее и поделить на std каждого спектра."""
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self
    def transform(self, X):
        X = np.asarray(X, float)
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction относительно среднего спектра train."""
    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, float).mean(0)
        return self
    def transform(self, X):
        X = np.asarray(X, float)
        out = np.empty_like(X)
        for i, row in enumerate(X):
            b, a = np.polyfit(self.ref_, row, 1)
            out[i] = (row - a) / (b if abs(b) > 1e-12 else 1.0)
        return out


class SavGol(BaseEstimator, TransformerMixin):
    def __init__(self, window=11, poly=2, deriv=0):
        self.window, self.poly, self.deriv = window, poly, deriv
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self
    def transform(self, X):
        return savgol_filter(np.asarray(X, float), self.window, self.poly, deriv=self.deriv, axis=1)


def _absorbance(X):
    X = np.clip(np.asarray(X, float), 1e-9, None)
    return -np.log10(X)


def to_absorbance():
    """-log10(T): для случаев, когда на входе пропускание/интенсивность в долях от опоры."""
    return FunctionTransformer(_absorbance, validate=False)


def get_preprocessor(name: str):
    """Названия пайплайнов предобработки, которые перебираем в эксперименте."""
    table = {
        "raw": lambda: FunctionTransformer(validate=False),
        "snv": lambda: SNV(),
        "msc": lambda: MSC(),
        "sg1": lambda: SavGol(11, 2, 1),
        "sg2": lambda: SavGol(11, 2, 2),
        "snv+sg1": lambda: make_pipeline(SNV(), SavGol(11, 2, 1)),
        "msc+sg1": lambda: make_pipeline(MSC(), SavGol(11, 2, 1)),
    }
    return table[name]()


PREPROCESSORS = ["raw", "snv", "msc", "sg1", "sg2", "snv+sg1", "msc+sg1"]

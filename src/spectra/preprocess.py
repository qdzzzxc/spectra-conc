"""Предобработка спектров как sklearn-трансформеры (работают построчно, без утечки между train/test,
кроме MSC, у которого эталон = средний спектр обучающей выборки)."""
from __future__ import annotations
import re
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, FunctionTransformer

# sg<deriv>[w<window>]: окно в точках; без w — 11 (как раньше)
_SG_RE = re.compile(r"^sg([12])(?:w(\d+))?$")


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
    """Фильтр Савицкого–Голея по оси длин волн: window — окно в точках (нечётное),
    poly — степень полинома, deriv — порядок производной."""

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


def _parse_sg(token: str) -> SavGol | None:
    """'sg1' -> SavGol(11, 2, 1), 'sg2w31' -> SavGol(31, 2, 2); не SG-токен -> None.
    Окно должно быть нечётным и > 2 (степень полинома), иначе ValueError."""
    m = _SG_RE.match(token)
    if not m:
        return None
    deriv = int(m.group(1))
    window = int(m.group(2)) if m.group(2) else 11
    if window % 2 == 0 or window <= 2:
        raise ValueError(f"Окно SG должно быть нечётным и больше 2: {token}")
    return SavGol(window, 2, deriv)


def get_preprocessor(name: str):
    """Предобработка по имени: токены raw/snv/msc/sg1/sg2, окно SG — суффикс wN (sg1w31),
    цепочка через "+" (snv+sg1w31)."""
    steps = []
    for token in name.split("+"):
        if token == "raw":
            step = FunctionTransformer(validate=False)
        elif token == "snv":
            step = SNV()
        elif token == "msc":
            step = MSC()
        else:
            step = _parse_sg(token)
            if step is None:
                raise KeyError(name)
        steps.append(step)
    return steps[0] if len(steps) == 1 else make_pipeline(*steps)


PREPROCESSORS = ["raw", "snv", "msc", "sg1", "sg2", "snv+sg1", "msc+sg1"]

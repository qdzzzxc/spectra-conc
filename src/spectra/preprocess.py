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

# osc<k>: OSC с k удаляемыми направлениями
_OSC_RE = re.compile(r"^osc(\d+)$")


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


class OSC(BaseEstimator, TransformerMixin):
    """Упрощённый OSC/EPO: убрать k главных направлений вариации X, ортогональной y (по train).
    Остаток R = Xc − yc·wᵀ, где w = Xcᵀyc / (ycᵀyc); направления — первые k правых сингулярных векторов R."""

    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("OSC требует y в fit")
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        yc = y - y.mean()
        w = Xc.T @ yc / (yc @ yc + 1e-12)
        R = Xc - np.outer(yc, w)
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Xc = X - self.mean_
        return X - (Xc @ self.components_.T) @ self.components_


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
    osc<k> — OSC с k направлениями (snv+osc2), цепочка через "+" (snv+sg1w31)."""
    steps = []
    for token in name.split("+"):
        if token == "raw":
            step = FunctionTransformer(validate=False)
        elif token == "snv":
            step = SNV()
        elif token == "msc":
            step = MSC()
        elif (m := _OSC_RE.match(token)):
            k = int(m.group(1))
            if k < 1:
                raise ValueError(f"Число компонент OSC должно быть >= 1: {token}")
            step = OSC(k)
        else:
            step = _parse_sg(token)
            if step is None:
                raise KeyError(name)
        steps.append(step)
    return steps[0] if len(steps) == 1 else make_pipeline(*steps)


PREPROCESSORS = ["raw", "snv", "msc", "sg1", "sg2", "snv+sg1", "msc+sg1"]

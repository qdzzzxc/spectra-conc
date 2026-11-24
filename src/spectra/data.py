"""Единый контейнер данных и простые загрузчики.

Любой датасет приводим к виду:
    X      (n_samples, n_features)  спектры / интенсивности по каналам
    wl     (n_features,)            длины волн (нм) или сдвиги (см^-1)
    y      (n_samples,)             целевая концентрация
    groups (n_samples,)             id физического образца/раствора (повторы одного раствора -> одна группа)
    meta   DataFrame                прочее: партия, день, температура, концентрации мешающих веществ...
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SpectraDataset:
    X: np.ndarray
    wl: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: str = "dataset"

    def __post_init__(self):
        assert self.X.ndim == 2
        assert len(self.X) == len(self.y) == len(self.groups)
        assert self.X.shape[1] == len(self.wl)

    def __len__(self):
        return len(self.y)

    def summary(self) -> str:
        return (
            f"{self.name}: n={len(self)}, p={self.X.shape[1]}, groups={len(np.unique(self.groups))}, "
            f"wl=[{self.wl.min():.1f},{self.wl.max():.1f}], y=[{self.y.min():.3g},{self.y.max():.3g}]"
        )


def load_wide_csv(path, y_col, group_col=None, meta_cols=(), name=None, sep=",") -> SpectraDataset:
    """CSV, где каждая строка - спектр, а колонки с числовыми именами - длины волн.

    Колонки, имя которых приводится к float, считаются каналами спектра.
    Если group_col не задан, каждая строка - отдельная группа (ОСТОРОЖНО: возможна утечка при повторах).
    """
    df = pd.read_csv(path, sep=sep)
    wl_cols, wl = [], []
    for c in df.columns:
        try:
            wl.append(float(c))
            wl_cols.append(c)
        except (TypeError, ValueError):
            pass
    X = df[wl_cols].to_numpy(float)
    y = df[y_col].to_numpy(float)
    groups = df[group_col].to_numpy() if group_col else np.arange(len(df))
    meta = df[list(meta_cols)].copy() if meta_cols else pd.DataFrame(index=df.index)
    return SpectraDataset(X, np.array(wl), y, groups, meta, name or str(path))

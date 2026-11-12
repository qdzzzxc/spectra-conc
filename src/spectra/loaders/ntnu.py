"""Загрузчик NTNU «Aqueous glucose measured by NIR spectroscopy» (DOI 10.18710/NSHFAK).

Единицы концентрации: мМ (glucose, lactate, acetaminophen, caffeine, ethanol).
Целевая переменная y = glucose, мМ.

ГРУППА = СОСТАВ раствора (строка из 5 концентраций). Один и тот же состав измерялся
в разных днях, run и кюветах (повторные приготовления/замеры одного рецепта), поэтому
для честного GroupKFold все спектры с одинаковым составом объединяются в одну группу —
иначе повторы одного раствора попадают одновременно в train и test (утечка).

Validation — отдельно смешанные растворы, отдельный день измерений (day 5).
Блоки (`block`) — подряд идущие повторы одного (состав, кювета, день, run); `repeat` —
номер повтора внутри блока.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import SpectraDataset

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "ntnu"
FILES = {"calibration": "CalibrationData.txt", "validation": "ValidationData.txt"}
INFO_COLS = {"Glucose (mM)": "glucose", "Lactate (mM)": "lactate", "Acetaminophen (mM)": "acetaminophen",
             "Caffeine (mM)": "caffeine", "Ethanol (mM)": "ethanol", "Temperature (C)": "temperature",
             "Kuvette": "cuvette", "Day": "day", "Run": "run"}
CONC_COLS = ["glucose", "lactate", "acetaminophen", "caffeine", "ethanol"]


def _read_raw(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Читает сырой файл сплита. Возвращает (info, wl, X).

    info — 9 служебных колонок, переименованных по INFO_COLS, индекс 0..n-1;
    wl — длины волн (нм); X — поглощение (n, n_channels). Неизвестный split -> KeyError.
    """
    df = pd.read_csv(RAW / FILES[split], sep="\t", encoding="utf-16")
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if not c.startswith("Column")]]
    info = df[list(INFO_COLS)].rename(columns=INFO_COLS).reset_index(drop=True)
    wl_cols = [c for c in df.columns if c not in INFO_COLS]
    wl = np.array([float(c) for c in wl_cols], dtype=float)
    X = df[wl_cols].to_numpy(dtype=float)
    return info, wl, X


def _add_structure(info: pd.DataFrame, split: str) -> pd.DataFrame:
    """Возвращает НОВЫЙ DataFrame с колонками set, composition, block, repeat."""
    out = info.copy()
    out["set"] = split
    out["composition"] = out[CONC_COLS].apply(lambda row: "|".join(f"{v:.2f}" for v in row), axis=1)
    key = out[["composition", "cuvette", "day", "run"]].astype(str).agg("||".join, axis=1)
    out["block"] = ((key != key.shift()).cumsum() - 1).astype(int)
    out["repeat"] = out.groupby("block").cumcount().astype(int)
    return out


def load(split: str = "calibration", wl_min: float | None = None, wl_max: float | None = None) -> SpectraDataset:
    """Загружает датасет NTNU как SpectraDataset.

    split ∈ {"calibration", "validation", "all"}; "all" — калибровка затем валидация
    (блоки валидации сдвигаются, чтобы не пересекались с калибровкой).
    wl_min/wl_max — обрезка диапазона длин волн. y = glucose (мМ),
    groups = фактор по составу раствора (см. docstring модуля).
    """
    if split == "all":
        parts: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
        for name in ("calibration", "validation"):
            info, wl, X = _read_raw(name)
            parts.append((_add_structure(info, name), wl, X))
        if not np.allclose(parts[0][1], parts[1][1]):
            raise ValueError("Длины волн в калибровке и валидации различаются")
        wl, X = parts[0][1], np.vstack([parts[0][2], parts[1][2]])
        meta = pd.concat([parts[0][0], parts[1][0]], ignore_index=True)
        meta.loc[meta["set"] == "validation", "block"] += int(parts[0][0]["block"].max()) + 1
    else:
        info, wl, X = _read_raw(split)
        meta = _add_structure(info, split)

    mask = np.ones(len(wl), dtype=bool)
    if wl_min is not None:
        mask &= wl >= wl_min
    if wl_max is not None:
        mask &= wl <= wl_max
    wl, X = wl[mask], X[:, mask]

    y = meta["glucose"].to_numpy(dtype=float)
    groups = pd.factorize(meta["composition"])[0]
    return SpectraDataset(X=X, wl=wl, y=y, groups=groups, meta=meta, name=f"ntnu_{split}")

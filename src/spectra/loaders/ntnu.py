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

from spectra.data import SpectraDataset

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "ntnu"
FILES = {"calibration": "CalibrationData.txt", "validation": "ValidationData.txt"}
INFO_COLS = {"Glucose (mM)": "glucose", "Lactate (mM)": "lactate", "Acetaminophen (mM)": "acetaminophen",
             "Caffeine (mM)": "caffeine", "Ethanol (mM)": "ethanol", "Temperature (C)": "temperature",
             "Kuvette": "cuvette", "Day": "day", "Run": "run"}
CONC_COLS = ["glucose", "lactate", "acetaminophen", "caffeine", "ethanol"]

# рабочие окна: без видимой области, насыщения ~1940 нм и шумного края >2400 нм
WORK_WINDOWS: list[tuple[float, float]] = [(1100.0, 1880.0), (2050.0, 2400.0)]

# валидация: метка 50 мМ, все модели предсказывают ~25 мМ — вероятная ошибка приготовления/метки в источнике
SUSPECT_COMPOSITIONS: tuple[str, ...] = ("50.00|15.00|0.15|0.10|34.50", "50.00|15.00|0.15|0.05|69.00")


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
    """Возвращает НОВЫЙ DataFrame с колонками set, composition, block, repeat, suspect."""
    out = info.copy()
    out["set"] = split
    out["composition"] = out[CONC_COLS].apply(lambda row: "|".join(f"{v:.2f}" for v in row), axis=1)
    key = out[["composition", "cuvette", "day", "run"]].astype(str).agg("||".join, axis=1)
    out["block"] = ((key != key.shift()).cumsum() - 1).astype(int)
    out["repeat"] = out.groupby("block").cumcount().astype(int)
    out["suspect"] = out["composition"].isin(SUSPECT_COMPOSITIONS)
    return out


def _resolve_windows(windows: list[tuple[float, float]] | str | None) -> list[tuple[float, float]] | None:
    """Приводит аргумент windows к списку кортежей (float, float).

    "work" — WORK_WINDOWS, другая строка -> ValueError, список списков (напр. из JSON) —
    к списку кортежей, None -> None.
    """
    if windows is None:
        return None
    if isinstance(windows, str):
        if windows != "work":
            raise ValueError(f"Неизвестное имя окон: {windows!r} (доступно только 'work')")
        return list(WORK_WINDOWS)
    return [(float(lo), float(hi)) for lo, hi in windows]


def load(
    split: str = "calibration",
    wl_min: float | None = None,
    wl_max: float | None = None,
    windows: list[tuple[float, float]] | str | None = None,
    drop_suspect: bool = False,
) -> SpectraDataset:
    """Загружает датасет NTNU как SpectraDataset.

    split ∈ {"calibration", "validation", "all"}; "all" — калибровка затем валидация
    (блоки валидации сдвигаются, чтобы не пересекались с калибровкой).
    wl_min/wl_max — обрезка диапазона длин волн.
    windows — окна длин волн: "work" (WORK_WINDOWS), список пар [lo, hi] (границы
    включительно, канал попадает если попал хотя бы в одно окно) или None;
    применяются вместе с wl_min/wl_max (логическое И).
    y = glucose (мМ), groups = фактор по составу раствора (см. docstring модуля).
    drop_suspect — исключить подозрительные составы (SUSPECT_COMPOSITIONS) и добавить
    суффикс "_nosus" к имени, если что-то исключено.
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
    resolved_windows = _resolve_windows(windows)
    if resolved_windows is not None:
        win_mask = np.zeros(len(wl), dtype=bool)
        for lo, hi in resolved_windows:
            win_mask |= (wl >= lo) & (wl <= hi)
        mask &= win_mask
    if not mask.any():
        raise ValueError("После фильтра по длинам волн не осталось ни одного канала")
    wl, X = wl[mask], X[:, mask]

    if drop_suspect:
        keep = ~meta["suspect"].to_numpy()
        n_dropped = int((~keep).sum())
        if n_dropped:
            meta = meta[keep].reset_index(drop=True)
            X = X[keep]
    else:
        n_dropped = 0

    y = meta["glucose"].to_numpy(dtype=float)
    groups = pd.factorize(meta["composition"])[0]
    name = f"ntnu_{split}_win" if resolved_windows is not None else f"ntnu_{split}"
    if n_dropped:
        name += "_nosus"
    return SpectraDataset(X=X, wl=wl, y=y, groups=groups, meta=meta, name=name)

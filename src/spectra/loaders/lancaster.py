"""Спектры Lancaster из экспорта Origin с проверкой исходных подписей."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import SpectraDataset

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data/raw/lancaster/origin_export"
SHEETS = {
    "aqueous_reflect": ("02_D/005_Raw - PLS", "03_C/001_water fixed 1"),
    "aqueous_transmit": (
        "04_HL2000TRlight/002_RAW - PLS", "04_HL2000TRlight/001_HL2000 TR light only 1",
    ),
    "intralipid_reflect": ("05_F/002_Raw - PLS", "05_F/001_Reflect 20pc IL eq vol water 5"),
    "intralipid_transmit": ("10_E/002_Raw - PLS", "10_E/001_TR 20pc IL eq vol water 5"),
}


def load(
    series: str, band: tuple[float, float] | None = None, root: str | Path = DEFAULT_ROOT
) -> SpectraDataset:
    """Сопоставляет сырые спектры с колонками Origin и группирует по концентрации."""
    root = Path(root)
    raw_name, source_name = SHEETS[series]
    raw = pd.read_csv(root / f"{raw_name}.csv")
    source = pd.read_csv(root / f"{source_name}.csv")
    metadata = pd.read_csv(root / f"{source_name}_columns.csv").set_index("csv_header")
    concentration = pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna()
    spectra = raw.loc[concentration.index, raw.columns[1:]].to_numpy(dtype=float)
    wavelengths = source.iloc[:, 0].to_numpy(dtype=float)
    columns = source.columns[1:]
    comments = metadata.loc[columns, "comments"].fillna("")
    labels = comments.str.extract(r"(\d+)mmol", flags=re.IGNORECASE, expand=False).astype(float)
    labels.loc[comments.str.contains(r"\bwater\b", case=False)] = 0
    candidates = source.loc[:, labels.dropna().index].to_numpy(dtype=float).T
    matches = np.isclose(spectra[:, None], candidates[None, :], rtol=0, atol=1e-9).all(axis=2)
    if not (matches.sum(axis=1) == 1).all() or (matches.sum(axis=0) > 1).any():
        raise ValueError(f"Lancaster {series}: не удалось однозначно сопоставить сырые спектры")
    source_indices = matches.argmax(axis=1)
    y = concentration.to_numpy(dtype=float)
    if not np.array_equal(y, labels.dropna().to_numpy()[source_indices]):
        raise ValueError(f"Lancaster {series}: метки концентрации расходятся")
    meta = pd.DataFrame({"source_comment": comments.loc[labels.dropna().index].to_numpy()[source_indices]})
    if band is not None:
        keep = (wavelengths >= band[0]) & (wavelengths <= band[1])
        if not keep.any():
            raise ValueError(f"Lancaster {series}: нет каналов в диапазоне {band}")
        spectra, wavelengths = spectra[:, keep], wavelengths[keep]
    return SpectraDataset(spectra, wavelengths, y, y.copy(), meta, name=f"lancaster_{series}")

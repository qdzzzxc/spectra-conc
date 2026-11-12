"""Тесты загрузчика NTNU."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "ntnu"
if not (RAW / "CalibrationData.txt").exists():
    pytest.skip("Нет данных data/raw/ntnu/ — пропуск тестов загрузчика NTNU", allow_module_level=True)

from spectra.loaders.ntnu import load  # noqa: E402


def test_calibration():
    ds = load("calibration")
    assert ds.X.shape == (273, 4200)
    assert ds.wl[0] == 400.0 and ds.wl[-1] == 2499.5
    assert np.isfinite(ds.X).all()
    assert ds.y.min() >= 0 and ds.y.max() <= 50
    assert len(np.unique(ds.groups)) == 56


def test_validation():
    ds = load("validation")
    assert ds.X.shape == (120, 4200)
    assert len(np.unique(ds.groups)) == 35
    assert (ds.meta["day"] == 5).all()


def test_all_no_group_leak():
    ds = load("all")
    assert len(ds) == 393
    assert len(np.unique(ds.groups)) == 91
    df = pd.DataFrame({"g": ds.groups, "set": ds.meta["set"]})
    both = df.groupby("g")["set"].nunique()
    assert (both == 1).all()


def test_structure():
    ds = load("calibration")
    meta = ds.meta
    assert meta.groupby("block")["composition"].nunique().max() == 1
    sizes = meta.groupby("block").size()
    assert set(sizes.unique()) <= {2, 3}
    assert (meta["repeat"] < 3).all()


def test_wl_range():
    ds = load("calibration", wl_min=1100, wl_max=2400)
    assert ((ds.wl >= 1100) & (ds.wl <= 2400)).all()
    assert ds.X.shape[1] == len(ds.wl)


def test_bad_split():
    with pytest.raises(KeyError):
        load("bad")


def test_windows_work():
    from spectra.loaders.ntnu import WORK_WINDOWS

    ds = load("calibration", windows="work")
    in_win = np.zeros(len(ds.wl), dtype=bool)
    for lo, hi in WORK_WINDOWS:
        in_win |= (ds.wl >= lo) & (ds.wl <= hi)
    assert in_win.all()
    assert not ((ds.wl > 1880) & (ds.wl < 2050)).any()
    assert ds.X.shape[1] == len(ds.wl)
    assert ds.name == "ntnu_calibration_win"


def test_windows_list_of_lists():
    ds = load("calibration", windows=[[1200, 1300]])
    assert ds.wl.min() == 1200.0 and ds.wl.max() == 1300.0
    assert len(ds.wl) == 201


def test_windows_bad():
    with pytest.raises(ValueError):
        load("calibration", windows="bad")
    with pytest.raises(ValueError):
        load("calibration", windows=[(3000, 4000)])


def test_suspect_flag():
    ds = load("validation")
    meta = ds.meta
    assert meta["suspect"].sum() == 6
    assert set(meta.loc[meta.suspect, "glucose"]) == {50.0}


def test_drop_suspect_validation():
    ds = load("validation", drop_suspect=True)
    assert len(ds) == 114
    assert "suspect" in ds.meta.columns and not ds.meta.suspect.any()
    assert ds.name.endswith("_nosus")
    assert ds.X.shape[0] == 114


def test_drop_suspect_calibration():
    ds = load("calibration", drop_suspect=True)
    assert len(ds) == 273
    assert not ds.name.endswith("_nosus")

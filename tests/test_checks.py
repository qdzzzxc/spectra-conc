import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from spectra.checks import (
    external_validation,
    interferent_exclusion,
    learning_curve,
    leave_one_day_out,
    subset,
)
from spectra.synthetic import make_synthetic


def _pair():
    cal = make_synthetic(n_solutions=20, seed=10)
    val = make_synthetic(n_solutions=10, seed=11)
    cal.meta["day"] = np.where(cal.groups < 10, 1, 2)
    return cal, val


def test_subset_consistency():
    cal, _ = _pair()
    mask = cal.groups % 2 == 0
    n = int(mask.sum())
    sub = subset(cal, mask)
    assert len(sub) == n and sub.X.shape[0] == n and len(sub.meta) == n


def test_external_validation():
    cal, val = _pair()
    res, pred = external_validation(cal, val, [("pls", "raw")])
    assert len(res) == 1 and res["r2"].iloc[0] > 0.9
    assert len(pred) == len(val)


def test_interferent_exclusion():
    cal, val = _pair()
    df = interferent_exclusion(cal, val, "interferent", [("pls", "raw")], zero_tol=30)
    assert len(df) == 2 and set(df["train_set"]) == {"without", "full"}
    assert np.isfinite(df["slope"]).all() and np.isfinite(df["slope_se"]).all()
    without = df[df["train_set"] == "without"]["n_train"].iloc[0]
    full = df[df["train_set"] == "full"]["n_train"].iloc[0]
    assert without < full


def test_leave_one_day_out():
    cal, _ = _pair()
    df = leave_one_day_out(cal, [("pls", "raw")])
    assert len(df) == 2 and set(df["day"]) == {1, 2}
    assert (df["n_dropped_groups"] == 0).all()


def test_learning_curve():
    cal, val = _pair()
    df = learning_curve(cal, val, [("pls", "raw")], sizes=[5, 50], seeds=[0, 1])
    assert len(df) == 4
    assert set(df["n_groups"]) <= {5, 20}

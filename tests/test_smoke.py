import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from spectra.synthetic import make_synthetic
from spectra.evaluate import nested_group_cv, run_grid
from spectra.metrics import regression_metrics
from spectra.preprocess import get_preprocessor, PREPROCESSORS


def test_preprocessors_shape():
    ds = make_synthetic(n_solutions=10)
    for p in PREPROCESSORS:
        out = get_preprocessor(p).fit(ds.X).transform(ds.X)
        assert out.shape == ds.X.shape and np.isfinite(out).all()


def test_group_cv_no_leak_and_quality():
    ds = make_synthetic(n_solutions=30, seed=1)
    oof, _ = nested_group_cv(ds, "pls", "raw", outer_splits=5)
    m = regression_metrics(ds.y, oof)
    assert m["r2"] > 0.9


def test_run_grid_runs():
    ds = make_synthetic(n_solutions=20, seed=2)
    res, oof = run_grid(ds, ["pls", "ridge"], ["raw", "snv"], do_extrapolation=True, verbose=False)
    assert (res.split == "group_cv").sum() == 4

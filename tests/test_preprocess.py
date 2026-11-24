import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from spectra.evaluate import nested_group_cv
from spectra.metrics import regression_metrics
from spectra.preprocess import OSC, SNV, SavGol, get_preprocessor
from spectra.synthetic import make_synthetic


def test_sg_window_from_name():
    sg1 = get_preprocessor("sg1")
    assert isinstance(sg1, SavGol) and (sg1.window, sg1.poly, sg1.deriv) == (11, 2, 1)
    sg2 = get_preprocessor("sg2w31")
    assert isinstance(sg2, SavGol) and sg2.window == 31 and sg2.deriv == 2


def test_chain_with_window():
    ds = make_synthetic(n_solutions=10)
    pipe = get_preprocessor("snv+sg1w21")
    assert isinstance(pipe, Pipeline) and len(pipe.steps) == 2
    second = pipe.steps[1][1]
    assert isinstance(second, SavGol) and second.window == 21
    out = pipe.fit(ds.X).transform(ds.X)
    assert out.shape == ds.X.shape and np.isfinite(out).all()


def test_matches_old_impl():
    ds = make_synthetic(n_solutions=10)
    got = get_preprocessor("snv+sg1").fit(ds.X).transform(ds.X)
    expected = SavGol(11, 2, 1).transform(SNV().transform(ds.X))
    assert np.allclose(got, expected)


def test_bad_names():
    for bad in ("sg1w20", "sg1w1"):
        with pytest.raises(ValueError):
            get_preprocessor(bad)
    for bad in ("foo", "snv+bar"):
        with pytest.raises(KeyError):
            get_preprocessor(bad)


def test_osc_components():
    ds = make_synthetic(n_solutions=20, seed=4)
    o = OSC(2).fit(ds.X, ds.y)
    assert o.components_.shape == (2, ds.X.shape[1])
    assert np.allclose(o.components_ @ o.components_.T, np.eye(2))


def test_osc_removes_directions():
    ds = make_synthetic(n_solutions=20, seed=4)
    o = OSC(2).fit(ds.X, ds.y)
    Xt = o.transform(ds.X)
    assert np.allclose((Xt - o.mean_) @ o.components_.T, 0, atol=1e-8)


def test_osc_needs_y():
    ds = make_synthetic(n_solutions=20, seed=4)
    with pytest.raises(ValueError):
        OSC(1).fit(ds.X)
    with pytest.raises(ValueError):
        get_preprocessor("osc0")


def test_osc_in_pipeline():
    ds = make_synthetic(n_solutions=20, seed=4)
    pipe = get_preprocessor("snv+osc2")
    assert isinstance(pipe, Pipeline) and len(pipe.steps) == 2
    second = pipe.steps[1][1]
    assert isinstance(second, OSC) and second.n_components == 2
    oof, _ = nested_group_cv(ds, "pls", "snv+osc2", outer_splits=4)
    assert regression_metrics(ds.y, oof)["r2"] > 0.9

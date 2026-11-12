import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from spectra.synthetic import make_synthetic
from spectra.preprocess import get_preprocessor, SavGol, SNV


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

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from sklearn.base import clone

from spectra.metrics import regression_metrics
from spectra.models import build
from spectra.synthetic import make_synthetic

torch = pytest.importorskip("torch")
from spectra.nn import CNN1DRegressor  # noqa: E402 — только после проверки torch


def test_cnn_fit_predict():
    ds = make_synthetic(n_solutions=20, seed=5)
    m = CNN1DRegressor(epochs=20).fit(ds.X, ds.y)
    p = m.predict(ds.X)
    assert p.shape == (len(ds),) and np.isfinite(p).all()


def test_cnn_deterministic():
    ds = make_synthetic(n_solutions=20, seed=5)
    p1 = CNN1DRegressor(epochs=10, random_state=3).fit(ds.X, ds.y).predict(ds.X)
    p2 = CNN1DRegressor(epochs=10, random_state=3).fit(ds.X, ds.y).predict(ds.X)
    assert np.allclose(p1, p2)


def test_cnn_clone():
    c = clone(CNN1DRegressor(epochs=5, augment=False))
    assert c.get_params()["epochs"] == 5
    assert c.get_params()["augment"] is False


def test_cnn_learns():
    ds = make_synthetic(n_solutions=30, seed=1)
    ds_test = make_synthetic(n_solutions=15, seed=2)
    m = CNN1DRegressor().fit(ds.X, ds.y)
    m_out = regression_metrics(ds_test.y, m.predict(ds_test.X))
    assert m_out["r2"] > 0.5


def test_build_cnn():
    pipe, grid = build("cnn", "snv", 100, n_train=50)
    assert grid == {"m__augment": [False, True]}
    assert isinstance(pipe.steps[-1][1], CNN1DRegressor)

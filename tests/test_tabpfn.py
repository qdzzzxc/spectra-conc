import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from spectra.synthetic import make_synthetic
from spectra.metrics import regression_metrics

pytest.importorskip("tabpfn")
WEIGHTS = pathlib.Path.home() / ".cache" / "tabpfn" / "tabpfn-v3.5-20260909.safetensors"

from spectra.models import build


def test_build_tabpfn():
    pipe, grid = build("tabpfn", "snv", 100, n_train=200)
    assert grid == {"pca__n_components": [10, 30]}
    assert [name for name, _ in pipe.steps] == ["prep", "pca", "m"]
    _, grid_small = build("tabpfn", "snv", 100, n_train=30)
    assert grid_small == {"pca__n_components": [10]}


@pytest.mark.skipif(not WEIGHTS.exists(), reason="нет весов TabPFN")
def test_tabpfn_fit_predict():
    ds = make_synthetic(n_solutions=20, seed=6)
    pipe, _ = build("tabpfn", "snv", ds.X.shape[1], n_train=len(ds))
    pipe.set_params(pca__n_components=10)
    pipe.fit(ds.X, ds.y)
    p = pipe.predict(ds.X)
    assert p.shape == (len(ds),)
    assert np.isfinite(p).all()
    assert regression_metrics(ds.y, p)["r2"] > 0.9

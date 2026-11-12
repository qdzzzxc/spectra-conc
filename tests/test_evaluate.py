import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from spectra.synthetic import make_synthetic
from spectra.evaluate import group_folds, run_grid
from spectra.models import _max_components


def test_group_folds_no_leak():
    ds = make_synthetic(n_solutions=12, seed=3)
    folds = group_folds(ds.groups, 5, seed=0)
    covered = []
    for tr, te in folds:
        assert set(ds.groups[tr]) & set(ds.groups[te]) == set()
        covered.extend(te.tolist())
    assert sorted(covered) == list(range(len(ds)))


def test_group_folds_seed():
    ds = make_synthetic(n_solutions=12, seed=3)
    f0a = group_folds(ds.groups, 5, seed=0)
    f0b = group_folds(ds.groups, 5, seed=0)
    for (_, te_a), (_, te_b) in zip(f0a, f0b):
        np.testing.assert_array_equal(te_a, te_b)
    f1 = group_folds(ds.groups, 5, seed=1)
    assert any(not np.array_equal(te_a, te_b) for (_, te_a), (_, te_b) in zip(f0a, f1))


def test_max_components_small():
    assert _max_components(400, 10) >= 1
    assert _max_components(400, 10) <= 4
    assert _max_components(400, 1000) == 15
    assert _max_components(3, 1000) == 3


def test_run_grid_fold_stats():
    ds = make_synthetic(n_solutions=15, seed=4)
    res, oof = run_grid(ds, ["pls"], ["raw"], outer_splits=3, do_extrapolation=False, verbose=False, seed=7)
    row = res[(res.split == "group_cv") & (res.model == "pls") & (res.prep == "raw")].iloc[0]
    std = row["rmse_fold_std"]
    assert np.isfinite(std) and std >= 0
    assert row["seed"] == 7
    assert set(oof["fold"].unique()) <= {0, 1, 2}


def test_extrapolation_split_with_ties():
    """Много групп на максимальной концентрации: тест экстраполяции не должен быть пустым."""
    from spectra.evaluate import extrapolation_split
    ds = make_synthetic(n_solutions=15, seed=5)
    top = np.isin(ds.groups, np.arange(5))
    ds.y[top] = ds.y.max()  # 5 из 15 групп на одном максимальном уровне -> квантиль 0.8 равен максимуму
    te, pred = extrapolation_split(ds, "pls", "raw")
    assert te.sum() > 0 and len(pred) == te.sum()
    assert set(ds.groups[te]) & set(ds.groups[~te]) == set()

"""Честная оценка: вложенная CV с разбиением по группам + проверка экстраполяции по концентрации."""
from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GridSearchCV

from .data import SpectraDataset
from .metrics import regression_metrics
from .models import build


def _fit_with_inner_cv(pipe, grid, X, y, groups, inner_splits=4):
    if not grid:
        return clone(pipe).fit(X, y)
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=max(2, min(inner_splits, n_groups)))
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


def nested_group_cv(ds: SpectraDataset, model: str, prep: str, outer_splits=5):
    """Внешний GroupKFold: повторы одного раствора никогда не попадают одновременно в train и test."""
    pipe, grid = build(model, prep, ds.X.shape[1])
    oof = np.full(len(ds), np.nan)
    n_groups = len(np.unique(ds.groups))
    for tr, te in GroupKFold(n_splits=min(outer_splits, n_groups)).split(ds.X, ds.y, ds.groups):
        est = _fit_with_inner_cv(pipe, grid, ds.X[tr], ds.y[tr], ds.groups[tr])
        oof[te] = est.predict(ds.X[te])
    return oof


def extrapolation_split(ds: SpectraDataset, model: str, prep: str, top_frac=0.2):
    """Обучаемся на нижних (1-top_frac) концентраций (по группам), тестируем на верхних.
    Показывает, умеет ли модель выходить за диапазон калибровки (деревья - нет)."""
    g = pd.DataFrame({"g": ds.groups, "y": ds.y}).groupby("g")["y"].mean()
    thr = g.quantile(1 - top_frac)
    test_groups = g.index[g > thr]
    te = np.isin(ds.groups, test_groups); tr = ~te
    pipe, grid = build(model, prep, ds.X.shape[1])
    est = _fit_with_inner_cv(pipe, grid, ds.X[tr], ds.y[tr], ds.groups[tr])
    pred = est.predict(ds.X[te])
    return te, pred


def run_grid(ds: SpectraDataset, models, preps, outer_splits=5, do_extrapolation=True, verbose=True):
    rows, oof_rows = [], []
    for prep in preps:
        for model in models:
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    oof = nested_group_cv(ds, model, prep, outer_splits)
                    m = regression_metrics(ds.y, oof)
                    row = {"model": model, "prep": prep, "split": "group_cv", **m, "sec": time.time() - t0}
                    rows.append(row)
                    for i, p in enumerate(oof):
                        oof_rows.append({"model": model, "prep": prep, "idx": i, "group": ds.groups[i], "y_true": ds.y[i], "y_pred": p})
                    if do_extrapolation:
                        te, pred = extrapolation_split(ds, model, prep)
                        rows.append({"model": model, "prep": prep, "split": "extrapolation",
                                     **regression_metrics(ds.y[te], pred), "sec": time.time() - t0})
            except Exception as e:  # не роняем весь прогон из-за одной комбинации
                rows.append({"model": model, "prep": prep, "split": "error", "error": repr(e)})
            if verbose:
                last = [r for r in rows if r["model"] == model and r["prep"] == prep and r["split"] == "group_cv"]
                print(f"{prep:>8} {model:>9}  " + (f"RMSE={last[0]['rmse']:.3f} R2={last[0]['r2']:.4f}" if last else "ERROR"))
    return pd.DataFrame(rows), pd.DataFrame(oof_rows)

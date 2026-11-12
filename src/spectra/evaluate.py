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


def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Внешние фолды по группам: GroupKFold с перемешиванием групп, детерминированно по seed."""
    n = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros((len(groups), 1)), groups=groups))


def _fit_with_inner_cv(pipe, grid, X, y, groups, inner_splits=4, seed=0):
    if not grid:
        return clone(pipe).fit(X, y)
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=max(2, min(inner_splits, n_groups)), shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


def nested_group_cv(ds: SpectraDataset, model: str, prep: str, outer_splits=5, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Внешний GroupKFold: повторы одного раствора никогда не попадают одновременно в train и test.
    Возвращает (oof-предсказания, номер внешнего фолда для каждого образца)."""
    oof = np.full(len(ds), np.nan)
    fold = np.full(len(ds), -1, dtype=int)
    for k, (tr, te) in enumerate(group_folds(ds.groups, outer_splits, seed)):
        pipe, grid = build(model, prep, ds.X.shape[1], n_train=len(tr))
        est = _fit_with_inner_cv(pipe, grid, ds.X[tr], ds.y[tr], ds.groups[tr], seed=seed)
        oof[te] = est.predict(ds.X[te])
        fold[te] = k
    return oof, fold


def extrapolation_split(ds: SpectraDataset, model: str, prep: str, top_frac=0.2, seed=0):
    """Обучаемся на нижних (1-top_frac) концентраций (по группам), тестируем на верхних.
    Показывает, умеет ли модель выходить за диапазон калибровки (деревья - нет)."""
    g = pd.DataFrame({"g": ds.groups, "y": ds.y}).groupby("g")["y"].mean()
    thr = g.quantile(1 - top_frac)
    test_groups = g.index[g > thr]
    te = np.isin(ds.groups, test_groups); tr = ~te
    pipe, grid = build(model, prep, ds.X.shape[1], n_train=int(tr.sum()))
    est = _fit_with_inner_cv(pipe, grid, ds.X[tr], ds.y[tr], ds.groups[tr], seed=seed)
    pred = est.predict(ds.X[te])
    return te, pred


def run_grid(ds: SpectraDataset, models, preps, outer_splits=5, do_extrapolation=True, verbose=True, seed=0):
    rows, oof_rows = [], []
    for prep in preps:
        for model in models:
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    oof, fold = nested_group_cv(ds, model, prep, outer_splits, seed=seed)
                    m = regression_metrics(ds.y, oof)
                    fold_rmses = [regression_metrics(ds.y[fold == f], oof[fold == f])["rmse"] for f in range(fold.max() + 1)]
                    rmse_fold_mean = float(np.mean(fold_rmses))
                    rmse_fold_std = float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else float("nan")
                    row = {"model": model, "prep": prep, "split": "group_cv", **m, "sec": time.time() - t0,
                           "seed": seed, "rmse_fold_mean": rmse_fold_mean, "rmse_fold_std": rmse_fold_std}
                    rows.append(row)
                    for i, p in enumerate(oof):
                        oof_rows.append({"model": model, "prep": prep, "idx": i, "group": ds.groups[i], "y_true": ds.y[i],
                                         "y_pred": p, "fold": int(fold[i]), "seed": seed})
                    if do_extrapolation:
                        te, pred = extrapolation_split(ds, model, prep, seed=seed)
                        rows.append({"model": model, "prep": prep, "split": "extrapolation", "seed": seed,
                                     **regression_metrics(ds.y[te], pred), "sec": time.time() - t0})
            except Exception as e:  # не роняем весь прогон из-за одной комбинации
                rows.append({"model": model, "prep": prep, "split": "error", "error": repr(e)})
            if verbose:
                last = [r for r in rows if r["model"] == model and r["prep"] == prep and r["split"] == "group_cv"]
                print(f"{prep:>8} {model:>9}  " + (f"RMSE={last[0]['rmse']:.3f} ±{last[0]['rmse_fold_std']:.3f} R2={last[0]['r2']:.4f}" if last else "ERROR"))
    return pd.DataFrame(rows), pd.DataFrame(oof_rows)

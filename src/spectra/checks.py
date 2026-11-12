"""Специальные проверки устойчивости: исключение интерферента, перенос на другой день, внешняя валидация, кривая обучения."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .data import SpectraDataset
from .evaluate import _fit_with_inner_cv
from .metrics import regression_metrics
from .models import build

Pipe = tuple[str, str]


def subset(ds: SpectraDataset, mask: np.ndarray) -> SpectraDataset:
    """Новый датасет из строк bool-маски: те же длины волн и имя, meta со сброшенным индексом."""
    return SpectraDataset(
        X=ds.X[mask],
        wl=ds.wl,
        y=ds.y[mask],
        groups=ds.groups[mask],
        meta=ds.meta[mask].reset_index(drop=True),
        name=ds.name,
    )


def fit_predict(train: SpectraDataset, X_test: np.ndarray, pipe: Pipe, seed: int = 0) -> np.ndarray:
    """Обучить пайплайн (с внутренним CV по гиперпараметрам) на train и предсказать X_test."""
    model, prep = pipe
    p, grid = build(model, prep, train.X.shape[1], n_train=len(train))
    est = _fit_with_inner_cv(p, grid, train.X, train.y, train.groups, seed=seed)
    return np.ravel(np.asarray(est.predict(X_test), dtype=float))


def _metrics_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Метрики регрессии, только rmse, mae, bias, r2, n."""
    m = regression_metrics(y, pred)
    return {k: m[k] for k in ("rmse", "mae", "bias", "r2", "n")}


def external_validation(cal: SpectraDataset, val: SpectraDataset, pipes: list[Pipe], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Внешняя валидация: обучение на всей калибровке, тест на всей валидации (другой день, другие составы).

    Возвращает (таблицу метрик по пайплайнам, предсказания по каждому спектру валидации).
    """
    rows, pred_rows = [], []
    for model, prep in pipes:
        pred = fit_predict(cal, val.X, (model, prep), seed=seed)
        rows.append({
            "model": model, "prep": prep,
            "n_train": len(cal), "n_train_groups": len(np.unique(cal.groups)),
            **_metrics_row(val.y, pred),
        })
        for idx in range(len(val)):
            pred_rows.append({"model": model, "prep": prep, "idx": idx, "y_true": float(val.y[idx]), "y_pred": float(pred[idx])})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def interferent_exclusion(
    cal: SpectraDataset,
    val: SpectraDataset,
    interferent: str,
    pipes: list[Pipe],
    seed: int = 0,
    zero_tol: float = 0.1,
) -> pd.DataFrame:
    """Дозовый отклик на интерферент: обучаемся без него (`without`) и на всей калибровке (`full`), тестируем на val.

    slope — на сколько мМ сдвигается предсказание глюкозы на 1 мМ интерферента
    (регрессия остатка pred - y_val по концентрации интерферента в val).
    """
    conc = cal.meta[interferent].to_numpy(dtype=float)
    rows = []
    for model, prep in pipes:
        for train_set in ("without", "full"):
            train = subset(cal, conc < zero_tol) if train_set == "without" else cal
            pred = fit_predict(train, val.X, (model, prep), seed=seed)
            res = pred - val.y
            lr = linregress(val.meta[interferent].to_numpy(dtype=float), res)
            rows.append({
                "interferent": interferent, "train_set": train_set, "model": model, "prep": prep,
                "n_train": len(train), "n_train_groups": len(np.unique(train.groups)),
                **_metrics_row(val.y, pred),
                "slope": lr.slope, "slope_se": lr.stderr, "slope_p": lr.pvalue,
            })
    return pd.DataFrame(rows)


def leave_one_day_out(cal: SpectraDataset, pipes: list[Pipe], seed: int = 0) -> pd.DataFrame:
    """Перенос между днями измерений: каждый день по очереди — тест, остальные дни — обучение.

    Из обучения исключаются группы, встречающиеся в тесте (иначе один и тот же состав
    окажется по обе стороны). Дни, где в обучении остаётся меньше 5 групп, пропускаются.
    """
    days = cal.meta["day"].to_numpy()
    rows = []
    for day in sorted(pd.unique(days)):
        te = days == day
        test_groups = np.unique(cal.groups[te])
        tr = ~te & ~np.isin(cal.groups, test_groups)
        n_dropped = int(((~te) & np.isin(cal.groups, test_groups)).sum())
        train_groups = np.unique(cal.groups[tr])
        if len(train_groups) < 5:
            continue
        train = subset(cal, tr)
        for model, prep in pipes:
            pred = fit_predict(train, cal.X[te], (model, prep), seed=seed)
            rows.append({
                "day": day, "model": model, "prep": prep,
                "n_train": len(train), "n_train_groups": len(train_groups), "n_dropped_groups": n_dropped,
                **_metrics_row(cal.y[te], pred),
            })
    return pd.DataFrame(rows)


def learning_curve(cal: SpectraDataset, val: SpectraDataset, pipes: list[Pipe], sizes: list[int], seeds: list[int]) -> pd.DataFrame:
    """Кривая «качество vs число обучающих составов»: для каждого размера — случайные группы калибровки.

    Выбор групп зависит только от пары (size, seed) и одинаков для всех пайплайнов.
    size больше числа доступных групп обрезается до числа групп.
    """
    ug = np.unique(cal.groups)
    rows = []
    for size in sizes:
        eff_size = min(size, len(ug))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(ug, size=eff_size, replace=False)
            train = subset(cal, np.isin(cal.groups, chosen))
            for model, prep in pipes:
                pred = fit_predict(train, val.X, (model, prep), seed=seed)
                rows.append({
                    "n_groups": eff_size, "seed": seed, "model": model, "prep": prep,
                    "n_train": len(train), **_metrics_row(val.y, pred),
                })
    return pd.DataFrame(rows)

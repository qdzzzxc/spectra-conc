"""Сравнение отдельных интервалов NTNU с PLS на всём рабочем диапазоне."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from spectra.data import SpectraDataset
from spectra.evaluate import group_folds
from spectra.loaders.ntnu import load
from spectra.metrics import regression_metrics
from spectra.models import build

LOGGER = logging.getLogger(__name__)
WINDOWS = ((1100, 1880), (2050, 2400))


def spectral_bands(wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    """Задаёт полосы шириной до 100 нм без пересечения через зону насыщения."""
    bands = {}
    for low, high in WINDOWS:
        for start in range(low, high, 100):
            end = min(start + 100, high)
            upper = wavelengths <= end if end == high else wavelengths < end
            bands[f"{start}–{end}"] = np.flatnonzero((wavelengths >= start) & upper)
    return bands


def select_channels(X: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Оставляет каналы выбранной полосы до расчёта SNV."""
    return X[:, columns]


def fit_pls(
    data: SpectraDataset, train: np.ndarray, bands: dict[str, np.ndarray], seed: int
) -> Pipeline:
    """Выбирает полосу и число PLS-компонент на внутренних групповых фолдах."""
    pipeline, _ = build("pls", "snv", data.X.shape[1], n_train=len(train))
    pipeline.steps.insert(0, ("band", FunctionTransformer(select_channels)))
    cv = group_folds(data.groups[train], 4, seed)
    min_inner_size = min(len(inner_train) for inner_train, _ in cv)
    grid = [
        {
            "band__kw_args": [{"columns": columns}],
            "m__n_components": list(range(1, min(15, len(columns), min_inner_size // 2) + 1)),
        }
        for columns in bands.values()
    ]
    search = GridSearchCV(
        pipeline, grid, cv=cv, scoring="neg_root_mean_squared_error",
        n_jobs=1, error_score="raise",
    )
    search.fit(data.X[train], data.y[train])
    return search.best_estimator_


def selected_band(model: Pipeline, bands: dict[str, np.ndarray]) -> str:
    """Возвращает подпись полосы, выбранной во внутренней CV."""
    columns = model.named_steps["band"].kw_args["columns"]
    return next(name for name, indices in bands.items() if np.array_equal(columns, indices))


def main() -> None:
    """Сохраняет ошибки фиксированных полос и вложенного выбора, включая внешнюю валидацию."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/ntnu_ipls"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    calibration = load("calibration", windows="work")
    validation = load("validation", windows="work", drop_suspect=True)
    if not np.array_equal(calibration.wl, validation.wl):
        raise ValueError("Шкалы длин волн калибровки и валидации отличаются")
    bands = spectral_bands(calibration.wl)
    full = {"Весь диапазон": np.arange(calibration.X.shape[1])}
    presets = {name: {name: columns} for name, columns in (bands | full).items()}
    presets["Вложенный выбор"] = bands
    rows, selections = [], []
    for seed in range(3):
        for name, candidates in presets.items():
            predictions = np.empty(len(calibration))
            for fold, (train, test) in enumerate(group_folds(calibration.groups, 5, seed)):
                model = fit_pls(calibration, train, candidates, seed)
                predictions[test] = model.predict(calibration.X[test])
                if name == "Вложенный выбор":
                    selections.append({"seed": seed, "fold": fold, "band": selected_band(model, bands)})
            score = regression_metrics(calibration.y, predictions)["rmse"]
            rows.append({"band": name, "split": "group_cv", "seed": seed, "rmse": score})
        LOGGER.info("Групповая CV завершена для seed %d", seed)

    train = np.arange(len(calibration))
    for name in ("Весь диапазон", "Вложенный выбор"):
        candidates = presets[name]
        model = fit_pls(calibration, train, candidates, seed=0)
        score = regression_metrics(validation.y, model.predict(validation.X))["rmse"]
        rows.append({
            "band": name, "split": "validation_clean", "seed": 0, "rmse": score,
            "selected_band": selected_band(model, candidates),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(f"{args.out}_results.csv", index=False)
    pd.DataFrame(selections).to_csv(f"{args.out}_selected.csv", index=False)
    cv = results[results["split"] == "group_cv"]
    summary = cv.groupby("band")["rmse"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    intervals = summary.loc[list(bands)]
    ax.bar(intervals.index, intervals["mean"], yerr=intervals["std"], capsize=3)
    ax.axhline(summary.loc["Весь диапазон", "mean"], color="tab:orange", label="PLS на всём диапазоне")
    ax.set(xlabel="Длина волны, нм", ylabel="RMSE, мМ", title="NTNU: PLS + SNV по спектральным полосам")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

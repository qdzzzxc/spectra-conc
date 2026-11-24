"""Единое сравнение спектральных моделей на трёх открытых датасетах."""

import argparse
import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from spectra.data import SpectraDataset
from spectra.evaluate import group_folds
from spectra.loaders import lancaster, ntnu
from spectra.metrics import regression_metrics
from spectra.models import MODELS, build
from spectra.preprocess import get_preprocessor

LOGGER = logging.getLogger(__name__)
COMPARISON_MODELS = (*MODELS, "cnn", "tabpfn", "catboost")


def load_cambridge(path: Path) -> SpectraDataset:
    """Читает лист Glucose, где концентрации записаны долями единицы."""
    labels, rows = [], []
    with ZipFile(path) as book:
        (sheet,) = [
            name
            for name in book.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        with book.open(sheet) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if not row.tag.endswith("}row"):
                    continue
                number = int(row.get("r"))
                values = [
                    (cell.get("r"), value)
                    for cell in row
                    if (value := cell.findtext("{*}v")) is not None
                ]
                if number == 2:
                    labels = [float(value) for cell, value in values if cell != "A2"]
                elif number >= 4 and values:
                    if len(values) != len(labels) + 1 or values[0][0] != f"A{number}":
                        raise ValueError(f"Неожиданные ячейки Cambridge в строке {number}")
                    rows.append([float(value) for _, value in values])
                row.clear()
    spectra = np.asarray(rows)
    y = np.asarray(labels) * 100
    if spectra.shape != (3925, len(y) + 1) or not np.isfinite(spectra).all():
        raise ValueError("Неожиданный размер или пропуски в спектрах Cambridge")
    return SpectraDataset(spectra[:, 1:].T, spectra[:, 0], y, y, name="cambridge")


def model_candidates(
    model: str,
    n_features: int,
    n_train: int,
    min_inner: int,
    device: str,
    sg: str,
) -> tuple[Pipeline, list[dict[str, list[Any]]]]:
    """Ограничивает вычисления и задаёт варианты предобработки и модели."""
    params: dict[str, list[Any]]
    if model == "catboost":
        from catboost import CatBoostRegressor

        estimator = CatBoostRegressor(
            iterations=150,
            learning_rate=0.08,
            border_count=32,
            rsm=0.2,
            thread_count=2,
            random_seed=0,
            verbose=False,
            allow_writing_files=False,
        )
        pipeline = Pipeline([("prep", "passthrough"), ("m", estimator)])
        params = {"m__depth": [3, 5]}
    else:
        pipeline, params = build(model, "raw", n_features, n_train)
    if model == "cnn":
        pipeline.set_params(m__device=device, m__epochs=80)
    if model == "gbm":
        pipeline.set_params(m__max_features=0.2, m__max_bins=32, m__min_samples_leaf=5)
        params = {"m__max_depth": [3, 5], "m__max_iter": [120]}
    if model == "svr":
        params = {"m__C": [10, 100], "m__gamma": ["scale", 0.01], "m__epsilon": [0.1]}
    if "pca" in pipeline.named_steps:
        pca = pipeline.named_steps["pca"]
        pca.random_state = 0
        if pca.n_components is not None:
            pca.n_components = min(pca.n_components, min_inner, n_features)
    for key in ("m__n_components", "pca__n_components"):
        if key in params:
            cap = min(n_features, min_inner)
            params[key] = sorted({min(value, cap) for value in params[key]})
    return pipeline, [{**params, "prep": [get_preprocessor(prep)]} for prep in ("raw", "snv", sg)]


def compare(
    data: SpectraDataset,
    model: str,
    splits: int,
    seed: int,
    device: str,
) -> dict[str, str | float | int]:
    """Оценивает модель на внешних фолдах после настройки на внутренних."""
    prediction = np.empty(len(data))
    fold_errors = []
    sg = "sg1w51" if data.name.startswith("ntnu") else "sg1"
    for train, test in group_folds(data.groups, splits, seed):
        inner = group_folds(data.groups[train], 4, seed)
        pipeline, grid = model_candidates(
            model,
            data.X.shape[1],
            len(train),
            min(len(i) for i, _ in inner),
            device,
            sg,
        )
        search = GridSearchCV(
            pipeline,
            grid,
            cv=inner,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        ).fit(data.X[train], data.y[train])
        prediction[test] = search.predict(data.X[test])
        fold_errors.append(regression_metrics(data.y[test], prediction[test])["rmse"])
    metrics = regression_metrics(data.y, prediction)
    return {
        "dataset": data.name,
        "model": model,
        "seed": seed,
        "unit": "п. п." if data.name == "cambridge" else "мМ",
        **metrics,
        "fold_rmse_std": float(np.std(fold_errors, ddof=1)),
    }


def main() -> None:
    """Запускает сравнение по всем моделям и сохраняет метрики каждого прогона."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/interim/model_comparison.csv"))
    parser.add_argument("--cnn-device", default="cpu")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("ntnu", "cambridge", "lancaster"),
        default=["ntnu", "cambridge", "lancaster"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--models", nargs="+", choices=COMPARISON_MODELS, default=COMPARISON_MODELS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = []
    if "ntnu" in args.datasets:
        datasets.append((ntnu.load("calibration", windows="work"), 5))
    if "cambridge" in args.datasets:
        datasets.append((load_cambridge(Path("data/raw/cambridge/glucose.xlsx")), 5))
    if "lancaster" in args.datasets:
        datasets.extend((lancaster.load(series), 7) for series in lancaster.SHEETS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for data, splits in datasets:
        for seed in args.seeds:
            for model in args.models:
                row = compare(data, model, splits, seed, args.cnn_device)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                LOGGER.info("%s seed=%d %s RMSE=%.3f", data.name, seed, model, row["rmse"])


if __name__ == "__main__":
    main()

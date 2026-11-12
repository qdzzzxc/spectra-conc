"""Специальные проверки устойчивости модели на датасете NTNU.

Запуск: uv run python scripts/run_ntnu_checks.py --quick --out reports/ntnu_checks
Результат: CSV по префиксу --out + PNG в reports/figures/.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from spectra.checks import external_validation, fit_predict, interferent_exclusion, leave_one_day_out, learning_curve, subset
from spectra.loaders.ntnu import load

FIGURES = pathlib.Path(__file__).resolve().parents[1] / "reports" / "figures"
DEFAULT_PIPES = ["pls:snv", "pls:sg1", "ridge:snv"]


def parse_pipes(raw: list[str]) -> list[tuple[str, str]]:
    """Разбирает аргументы вида model:prep в список пар (model, prep)."""
    pipes: list[tuple[str, str]] = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != 2 or not all(parts):
            raise SystemExit(f"--pipes: ожидалось model:prep, получено {item!r}")
        pipes.append((parts[0], parts[1]))
    return pipes


def pipe_label(pipe: tuple[str, str]) -> str:
    """Подпись пайплайна в легенде: prep | model."""
    model, prep = pipe
    return f"{prep} | {model}"


def _step(name: str) -> float:
    print(f"\n=== {name} ===", flush=True)
    return time.time()


def _done(t0: float) -> None:
    print(f"готово за {time.time() - t0:.1f} с", flush=True)


def _table(df: pd.DataFrame) -> str:
    return df.round(3).to_string(index=False)


def plot_learning_curve(lc: pd.DataFrame, pipes: list[tuple[str, str]]) -> pathlib.Path:
    """Кривая обучения: RMSEP на валидации против числа обучающих составов, среднее по seed и полоса ±std."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for pipe in pipes:
        sub = lc[(lc["model"] == pipe[0]) & (lc["prep"] == pipe[1])]
        agg = sub.groupby("n_groups")["rmse"].agg(["mean", "std"])
        x = agg.index.to_numpy(float)
        mean = agg["mean"].to_numpy()
        std = np.nan_to_num(agg["std"].to_numpy(), nan=0.0)
        ax.plot(x, mean, marker="o", label=pipe_label(pipe))
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("число составов в обучении, шт")
    ax.set_ylabel("RMSEP на валидации, мМ")
    ax.set_title("Кривая обучения: качество от числа обучающих составов (NTNU)")
    ax.legend(fontsize=8)
    path = FIGURES / "ntnu_learning_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_interferents(cal, val, interferents: list[str], pipe: tuple[str, str], seed: int) -> pathlib.Path:
    """Остаток предсказания против концентрации интерферента для обучения без него и на всей калибровке."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"without": "tab:blue", "full": "tab:red"}
    fig, axes = plt.subplots(1, len(interferents), figsize=(5 * len(interferents), 4.5), squeeze=False)
    for ax, interf in zip(axes[0], interferents):
        conc_val = val.meta[interf].to_numpy(dtype=float)
        conc_cal = cal.meta[interf].to_numpy(dtype=float)
        for train_set in ("without", "full"):
            train = subset(cal, conc_cal < 0.1) if train_set == "without" else cal
            pred = fit_predict(train, val.X, pipe, seed=seed)
            res = pred - val.y
            lr = linregress(conc_val, res)
            xs = np.array([conc_val.min(), conc_val.max()], dtype=float)
            ax.scatter(conc_val, res, s=12, color=colors[train_set], alpha=0.6)
            ax.plot(xs, lr.intercept + lr.slope * xs, color=colors[train_set], lw=1.5,
                    label=f"{train_set}: {lr.slope:.2f} мМ/мМ")
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(f"{interf}, мМ")
        ax.set_ylabel("предсказание − глюкоза, мМ")
        ax.set_title(f"Интерферент: {interf}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Дозовый отклик на интерференты (пайплайн {pipe_label(pipe)})")
    path = FIGURES / "ntnu_interferents.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_lodo(lodo: pd.DataFrame, pipes: list[tuple[str, str]]) -> pathlib.Path:
    """RMSE по дням (leave-one-day-out), группировка столбцов по пайплайну."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    days = sorted(pd.unique(lodo["day"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(pipes))
    for i, pipe in enumerate(pipes):
        sub = lodo[(lodo["model"] == pipe[0]) & (lodo["prep"] == pipe[1])].set_index("day")
        vals = [float(sub.loc[d, "rmse"]) if d in sub.index else np.nan for d in days]
        ax.bar(np.arange(len(days)) + i * width, vals, width=width, label=pipe_label(pipe))
    ax.set_xticks(np.arange(len(days)) + width * (len(pipes) - 1) / 2)
    ax.set_xticklabels([f"день {d}" for d in days])
    ax.set_xlabel("день измерений (исключён из обучения)")
    ax.set_ylabel("RMSE, мМ")
    ax.set_title("Leave-one-day-out: перенос между днями")
    ax.legend(fontsize=8)
    path = FIGURES / "ntnu_lodo.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    """Прогон всех специальных проверок устойчивости + графики."""
    ap = argparse.ArgumentParser(description="Специальные проверки устойчивости (NTNU)")
    ap.add_argument("--out", default="reports/ntnu_checks", help="префикс выходных файлов")
    ap.add_argument("--pipes", nargs="*", default=None, help="пайплайны вида model:prep")
    ap.add_argument("--quick", action="store_true", help="быстрый прогон: один пайплайн, мало размеров/seed")
    ap.add_argument("--seed", type=int, default=0, help="seed для внешней валидации, интерферентов и LODO")
    a = ap.parse_args()

    pipes_explicit = a.pipes is not None
    pipes = parse_pipes(a.pipes if pipes_explicit else (["pls:snv"] if a.quick else DEFAULT_PIPES))
    sizes = [10, 30, 56] if a.quick else [6, 10, 15, 20, 30, 40, 56]
    seeds = [0, 1] if a.quick else [0, 1, 2, 3, 4]

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = _step("Загрузка данных (windows='work')")
    cal = load("calibration", windows="work")
    val = load("validation", windows="work")
    print(cal.summary())
    print(val.summary())
    _done(t0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        t0 = _step("Внешняя валидация (обучение на калибровке, тест на валидации)")
        ext, ext_pred = external_validation(cal, val, pipes, seed=a.seed)
        ext.to_csv(f"{out}_external.csv", index=False)
        ext_pred.to_csv(f"{out}_external_pred.csv", index=False)
        print(_table(ext))
        _done(t0)

        t0 = _step("Исключение интерферента (lactate, ethanol)")
        interf = pd.concat([interferent_exclusion(cal, val, name, pipes, seed=a.seed)
                            for name in ("lactate", "ethanol")], ignore_index=True)
        interf.to_csv(f"{out}_interferents.csv", index=False)
        print(_table(interf))
        _done(t0)

        t0 = _step("Leave-one-day-out")
        lodo = leave_one_day_out(cal, pipes, seed=a.seed)
        lodo.to_csv(f"{out}_lodo.csv", index=False)
        print(_table(lodo))
        _done(t0)

        t0 = _step(f"Кривая обучения (sizes={sizes}, seeds={seeds})")
        lc = learning_curve(cal, val, pipes, sizes=sizes, seeds=seeds)
        lc.to_csv(f"{out}_learning.csv", index=False)
        print(_table(lc))
        _done(t0)

    t0 = _step("Графики")
    print(plot_learning_curve(lc, pipes))
    print(plot_interferents(cal, val, ["lactate", "ethanol"], pipes[0], a.seed))
    print(plot_lodo(lodo, pipes))
    _done(t0)


if __name__ == "__main__":
    main()

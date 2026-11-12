"""Проверка пригодности датасета Mendeley Raman (DOI 10.17632/yk75sxk6p5.1).

Есть ли в спектрах сигнал концентрации при честном сплите «уровень целиком в тесте»
(LeaveOneGroupOut по уровню концентрации) и чем он отличается от перемешанных меток.

Запуск: uv run python scripts/check_mendeley_raman.py
Результат: таблица в stdout + reports/figures/mendeley_raman_means.png.
"""
from __future__ import annotations

import pathlib
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mendeley_raman"
SETS = ("Fourteenlabels", "Tenlabels")
N_COMPONENTS = (1, 2, 3, 5)


def read_set(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Спектры, числовая метка (единицы в источнике не определены), id уровня, исходные подписи."""
    X = pd.read_csv(RAW / f"{name}_RamanData.csv", header=None).to_numpy(float)
    lab = pd.read_csv(RAW / f"{name}_labels.csv")
    raw_labels = lab["Label name"].astype(str).tolist()
    y = np.array([float(re.sub(r"[^0-9.]", "", s)) for s in raw_labels])
    return X, y, lab["has_DM2"].to_numpy(), raw_labels


def snv(X: np.ndarray) -> np.ndarray:
    """SNV по строкам (без обучаемых параметров, утечки нет)."""
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def logo_rmse(X: np.ndarray, y: np.ndarray, g: np.ndarray, n_comp: int) -> tuple[float, float]:
    """RMSE и корреляция PLS при исключении уровня концентрации целиком."""
    p = cross_val_predict(PLSRegression(n_comp, scale=False), X, y, groups=g, cv=LeaveOneGroupOut()).ravel()
    return float(np.sqrt(np.mean((p - y) ** 2))), float(np.corrcoef(p, y)[0, 1])


def check_set(name: str, seed: int = 0) -> pd.DataFrame:
    """Таблица RMSE для raw/SNV × число компонент + бейзлайн с перемешанными по уровням метками."""
    X, y, g, raw_labels = read_set(name)
    print(f"{name}: {X.shape[0]} спектров × {X.shape[1]} каналов, уровни: {sorted(set(raw_labels), key=raw_labels.index)}")
    rng = np.random.default_rng(seed)
    levels = np.unique(g)
    perm = dict(zip(levels, rng.permutation([y[g == k][0] for k in levels])))
    y_perm = np.array([perm[k] for k in g])
    rows = []
    for prep, Z in (("raw", X), ("snv", snv(X))):
        for nc in N_COMPONENTS:
            rmse, r = logo_rmse(Z, y, g, nc)
            rmse_perm, _ = logo_rmse(Z, y_perm, g, nc)
            rows.append({"set": name, "prep": prep, "n_comp": nc, "rmse": rmse, "r": r,
                         "rmse_permuted": rmse_perm, "std_y": float(y.std())})
    return pd.DataFrame(rows)


def plot_means() -> None:
    """Средний SNV-спектр каждого уровня (ось X — номер канала: шкала в источнике не дана)."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(12, 4))
    for ax, name in zip(axes, SETS):
        X, y, g, _ = read_set(name)
        Z = snv(X)
        cmap = plt.get_cmap("viridis")
        for k in np.unique(g):
            v = y[g == k][0]
            ax.plot(Z[g == k].mean(0), color=cmap((v - y.min()) / np.ptp(y)), lw=0.8, label=f"{v:g}")
        ax.set(title=name, xlabel="номер канала", ylabel="SNV")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Mendeley Raman: средний спектр уровня; цвет — число из подписи (единицы в источнике не определены)")
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "mendeley_raman_means.png"
    fig.savefig(out, dpi=120)
    print(f"сохранено: {out.relative_to(ROOT)}")


def main() -> None:
    res = pd.concat([check_set(n) for n in SETS], ignore_index=True)
    print(res.round(3).to_string(index=False))
    plot_means()


if __name__ == "__main__":
    main()

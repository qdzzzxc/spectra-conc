"""Разведочные графики по датасету NTNU (NIR глюкоза).

Запуск: uv run python scripts/explore_ntnu.py
Результат: PNG в reports/figures/ + reports/figures/ntnu_outliers.csv.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from spectra.loaders.ntnu import load
from spectra.preprocess import SNV

OUT = pathlib.Path(__file__).resolve().parents[1] / "reports" / "figures"
CMAP = plt.get_cmap("viridis")
NORM = Normalize(0, 50)


def _mask(ds, lo: float, hi: float) -> np.ndarray:
    return (ds.wl >= lo) & (ds.wl <= hi)


def _save(fig: matplotlib.figure.Figure, name: str) -> pathlib.Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_spectra_full(ds) -> pathlib.Path:
    """Все спектры калибровки; серые зоны — сильное поглощение воды."""
    cal = ds.meta["set"].to_numpy() == "calibration"
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in np.flatnonzero(cal):
        ax.plot(ds.wl, ds.X[i], color=CMAP(NORM(ds.y[i])), lw=0.5)
    for lo, hi in [(1880, 2050), (2400, 2500)]:
        ax.axvspan(lo, hi, color="gray", alpha=0.15)
    ax.set_xlabel("длина волны, нм")
    ax.set_ylabel("поглощение, отн. ед.")
    handles = [Line2D([0], [0], color="gray", alpha=0.4, lw=6, label="сильное поглощение воды / возможное насыщение")]
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=ax)
    cb.set_label("глюкоза, мМ")
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    return _save(fig, "ntnu_spectra_full.png")


def plot_spectra_diff(ds) -> pathlib.Path:
    """Спектры калибровки минус средний спектр, два поддиапазона."""
    cal = ds.meta["set"].to_numpy() == "calibration"
    mean_spec = ds.X[cal].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (lo, hi) in zip(axes, [(1100, 1880), (2050, 2400)]):
        m = _mask(ds, lo, hi)
        for i in np.flatnonzero(cal):
            ax.plot(ds.wl[m], ds.X[i][m] - mean_spec[m], color=CMAP(NORM(ds.y[i])), lw=0.5)
        ax.set_xlabel("длина волны, нм")
        ax.set_ylabel("поглощение − среднее, отн. ед.")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axes.tolist(), shrink=0.8)
    cb.set_label("глюкоза, мМ")
    return _save(fig, "ntnu_spectra_diff.png")


def plot_repeats(ds) -> pathlib.Path:
    """Разброс повторов внутри блоков vs разброс между блоками."""
    cal = ds.meta["set"].to_numpy() == "calibration"
    m = _mask(ds, 1100, 2400)
    X_sub = ds.X[:, m]
    meta = ds.meta.loc[ds.meta["set"] == "calibration"]
    idx = np.flatnonzero(cal)
    block_means = []
    block_stds = []
    for b, grp in meta.groupby("block"):
        rows = grp.index.to_numpy()
        Xb = X_sub[rows]
        block_means.append(Xb.mean(axis=0))
        if len(rows) > 1:
            block_stds.append(Xb.std(axis=0, ddof=1))
    within = np.median(np.vstack(block_stds), axis=0)
    between = np.vstack(block_means).std(axis=0, ddof=1)
    wl_sub = ds.wl[m]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wl_sub, within, label="внутри блока (повторы)")
    ax.plot(wl_sub, between, label="между блоками")
    ax.set_yscale("log")
    ax.set_xlabel("длина волны, нм")
    ax.set_ylabel("стандартное отклонение, отн. ед. (лог)")
    ax.legend()
    return _save(fig, "ntnu_repeats.png")


def plot_drift(ds) -> pathlib.Path:
    """Дрейф среднего поглощения и температуры по порядковому номеру спектра."""
    m = _mask(ds, 1100, 1300)
    mean_abs = ds.X[:, m].mean(axis=1)
    days = ds.meta["day"].to_numpy()
    uniq_days = np.unique(days)
    colors = {d: plt.get_cmap("tab10")(k % 10) for k, d in enumerate(uniq_days)}
    pts_color = [colors[d] for d in days]
    order = np.arange(len(ds))
    n_cal = int((ds.meta["set"] == "calibration").sum())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.scatter(order, mean_abs, c=pts_color, s=8)
    ax2.scatter(order, ds.meta["temperature"].to_numpy(), c=pts_color, s=8)
    for ax in (ax1, ax2):
        ax.axvline(n_cal - 0.5, color="black", ls="--", lw=1)
    ax1.set_ylabel("среднее поглощение\n1100–1300 нм, отн. ед.")
    ax2.set_xlabel("порядковый номер спектра")
    ax2.set_ylabel("температура, °C")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6, label=f"день {d}")
               for d, c in colors.items()]
    ax1.legend(handles=handles, fontsize=8, loc="upper left", ncols=3)
    return _save(fig, "ntnu_drift.png")


def _pca_data(ds):
    """SNV на 1100–1880 нм (fit построчный, утечки нет). Возвращает (wl_mask, Xs)."""
    m = _mask(ds, 1100, 1880)
    Xs = SNV().fit_transform(ds.X[:, m])
    return m, Xs


def plot_pca(ds) -> pathlib.Path:
    """PC1 vs PC2; PCA обучен на калибровке; цвет — день, маркер — set."""
    cal = (ds.meta["set"] == "calibration").to_numpy()
    _, Xs = _pca_data(ds)
    pca = PCA(n_components=2, random_state=0).fit(Xs[cal])
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_ * 100
    days = ds.meta["day"].to_numpy()
    uniq_days = np.unique(days)
    colors = {d: plt.get_cmap("tab10")(k % 10) for k, d in enumerate(uniq_days)}
    fig, ax = plt.subplots(figsize=(10, 5))
    for d in uniq_days:
        sel = days == d
        marker = "o" if cal[sel][0] else "^"
        ax.scatter(scores[sel, 0], scores[sel, 1], c=[colors[d]], marker=marker, s=14, label=f"день {d}")
    ax.set_xlabel(f"PC1, {evr[0]:.1f}%")
    ax.set_ylabel(f"PC2, {evr[1]:.1f}%")
    ax.legend(fontsize=8)
    return _save(fig, "ntnu_pca.png")


def find_outliers(ds) -> tuple[pathlib.Path, pd.DataFrame]:
    """Выбросы по невязке реконструкции 5-компонентного PCA (fit на калибровке)."""
    cal = (ds.meta["set"] == "calibration").to_numpy()
    _, Xs = _pca_data(ds)
    pca = PCA(n_components=5, random_state=0).fit(Xs[cal])
    x_hat = pca.inverse_transform(pca.transform(Xs))
    q = ((Xs - x_hat) ** 2).sum(axis=1)
    q_cal = q[cal]
    med = np.median(q_cal)
    mad = np.median(np.abs(q_cal - med))
    threshold = med + 5 * mad
    out = pd.DataFrame({
        "idx": np.arange(len(ds)),
        "set": ds.meta["set"],
        "block": ds.meta["block"],
        "repeat": ds.meta["repeat"],
        "glucose": ds.y,
        "day": ds.meta["day"],
        "run": ds.meta["run"],
        "cuvette": ds.meta["cuvette"],
        "temperature": ds.meta["temperature"],
        "q": q,
        "is_outlier": q > threshold,
    }).sort_values("q", ascending=False).reset_index(drop=True)
    path = OUT / "ntnu_outliers.csv"
    out.to_csv(path, index=False)
    print(f"порог q = {threshold:.4g} (медиана калибровки {med:.4g} + 5*MAD {mad:.4g})")
    for name in ["calibration", "validation"]:
        sub = out[out["set"] == name]
        print(f"выбросов [{name}]: {int(sub['is_outlier'].sum())} из {len(sub)}")
    print("топ-10 по q:")
    print(out.head(10).to_string(index=False))
    return path, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load("all")
    print(ds.summary())
    paths = [
        plot_spectra_full(ds),
        plot_spectra_diff(ds),
        plot_repeats(ds),
        plot_drift(ds),
        plot_pca(ds),
    ]
    csv_path, _ = find_outliers(ds)
    paths.append(csv_path)
    print("\nсохранено:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()

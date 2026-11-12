"""Синтетические спектры по закону Бугера-Ламберта для отладки пайплайна ДО появления реальных данных.

A(wl) = l * (c * eps_analyte(wl) + sum_k c_k * eps_k(wl)) + baseline + рассеяние + шум
Параметры шума/мешающих веществ можно ломать, чтобы проверять устойчивость моделей.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import SpectraDataset


def _gauss(wl, mu, sig): return np.exp(-0.5 * ((wl - mu) / sig) ** 2)


def make_synthetic(n_solutions=40, repeats=3, wl_min=1400, wl_max=2200, step=2.0,
                   c_max=400.0, interferent=True, noise=2e-3, seed=0) -> SpectraDataset:
    rng = np.random.default_rng(seed)
    wl = np.arange(wl_min, wl_max + step, step)
    eps_a = 0.9 * _gauss(wl, 1690, 25) + 0.6 * _gauss(wl, 2100, 30) + 0.4 * _gauss(wl, 2270, 20)
    eps_i = 0.8 * _gauss(wl, 1750, 40) + 0.5 * _gauss(wl, 2000, 35)
    eps_w = 1.5 * _gauss(wl, 1450, 60) + 1.2 * _gauss(wl, 1940, 50)
    c = rng.uniform(0, c_max, n_solutions)
    ci = rng.uniform(0, 100, n_solutions) if interferent else np.zeros(n_solutions)
    rows, y, groups, meta = [], [], [], []
    for s in range(n_solutions):
        for r in range(repeats):
            T = rng.normal(25, 0.5)                             # температура сдвигает полосу воды
            water = 1.0 * np.roll(eps_w, int(round((T - 25) * 1.0)))
            scatter = rng.normal(0, 0.01) + rng.normal(0, 0.005) * (wl - wl.mean()) / 400
            a = 1e-3 * (c[s] * eps_a + ci[s] * eps_i) + water + scatter
            a += rng.normal(0, noise, len(wl))
            rows.append(a); y.append(c[s]); groups.append(s)
            meta.append({"interferent": ci[s], "temp": T, "repeat": r})
    return SpectraDataset(np.array(rows), wl, np.array(y), np.array(groups), pd.DataFrame(meta), "synthetic")

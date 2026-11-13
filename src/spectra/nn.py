"""Нейросетевые модели на torch (CPU) с интерфейсом sklearn.

Требуют группу зависимостей nn: uv sync --group nn.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin


def _make_net(n_filters: int, kernel_size: int, hidden: int, dropout: float) -> nn.Sequential:
    """AvgPool(4) -> [Conv1d -> ReLU -> AvgPool(4)] x 2 -> Flatten -> Dropout -> Linear(hidden) -> ReLU -> Linear(1).

    Первый AvgPool(4) усредняет соседние каналы: спектр передискретизирован, а обучение на CPU в 4 раза быстрее.
    """
    pad = kernel_size // 2
    return nn.Sequential(
        nn.AvgPool1d(4),
        nn.Conv1d(1, n_filters, kernel_size, padding=pad), nn.ReLU(), nn.AvgPool1d(4),
        nn.Conv1d(n_filters, n_filters, kernel_size, padding=pad), nn.ReLU(), nn.AvgPool1d(4),
        nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(hidden), nn.ReLU(), nn.Linear(hidden, 1),
    )


class CNN1DRegressor(BaseEstimator, RegressorMixin):
    """1D-CNN по спектру.

    Вход центрируется средним спектром train и делится на общий std (одно число); y стандартизуется.
    Аугментация (только при обучении): случайные смещение, наклон базовой линии, шум и сдвиг по оси на
    ±shift_max каналов.
    """

    def __init__(self, n_filters=8, kernel_size=9, hidden=32, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 epochs=200, batch_size=64, augment=True, offset_std=0.1, slope_std=0.1, noise_std=0.02,
                 shift_max=2, random_state=0):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.augment = augment
        self.offset_std = offset_std
        self.slope_std = slope_std
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.random_state = random_state

    def fit(self, X, y):
        """Обучить сеть на спектрах X (n, p) и концентрациях y."""
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32).ravel()
        self.x_mean_ = X.mean(0)
        Xc = X - self.x_mean_
        self.x_scale_ = float(Xc.std()) or 1.0
        Xn = Xc / self.x_scale_
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std()) or 1.0
        yn = (y - self.y_mean_) / self.y_scale_
        self.net_ = _make_net(self.n_filters, self.kernel_size, self.hidden, self.dropout)
        self.net_(torch.zeros(1, 1, X.shape[1]))
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt = torch.from_numpy(Xn).unsqueeze(1)
        yt = torch.from_numpy(yn)
        self.net_.train()
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = Xt[idx]
                if self.augment:
                    xb = self._augment(xb, rng)
                loss = torch.mean((self.net_(xb).squeeze(1) - yt[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.n_features_in_ = X.shape[1]
        return self

    def _augment(self, xb: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Аугментация батча: смещение, наклон базовой линии, шум и сдвиг по оси."""
        b, _, p = xb.shape
        t = torch.linspace(-1, 1, p)
        offset = torch.from_numpy(rng.normal(0, self.offset_std, (b, 1, 1)).astype(np.float32))
        slope = torch.from_numpy(rng.normal(0, self.slope_std, (b, 1, 1)).astype(np.float32))
        noise = torch.from_numpy(rng.normal(0, self.noise_std, (b, 1, p)).astype(np.float32))
        out = xb + offset + slope * t + noise
        if self.shift_max > 0:
            s = int(rng.integers(-self.shift_max, self.shift_max + 1))
            out = torch.roll(out, s, dims=2)
        return out

    def predict(self, X):
        """Предсказать концентрации для спектров X."""
        Xn = (np.asarray(X, np.float32) - self.x_mean_) / self.x_scale_
        self.net_.eval()
        with torch.no_grad():
            p = self.net_(torch.from_numpy(Xn).unsqueeze(1)).squeeze(1).numpy()
        return p.astype(float) * self.y_scale_ + self.y_mean_

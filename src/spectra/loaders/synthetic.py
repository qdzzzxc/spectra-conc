"""Загрузчик синтетики для проверки CLI без реальных данных."""

from spectra.data import SpectraDataset
from spectra.synthetic import make_synthetic


def load() -> SpectraDataset:
    """Синтетический датасет с параметрами по умолчанию."""
    return make_synthetic()

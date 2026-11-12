"""Загрузчик синтетики для проверки CLI без реальных данных."""
from ..data import SpectraDataset
from ..synthetic import make_synthetic


def load() -> SpectraDataset:
    """Синтетический датасет с параметрами по умолчанию."""
    return make_synthetic()

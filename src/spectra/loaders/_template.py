"""ШАБЛОН загрузчика. Скопируйте в ntnu.py / mendeley_raman.py и заполните после изучения файлов в data/raw/.

Обязательно определить:
  - что является ОДНИМ физическим образцом (groups) - повторы одного раствора должны иметь одинаковый group;
  - что является целевой концентрацией y (и в каких единицах);
  - какие мета-поля нужны для проверок (партия, день, температура, концентрации мешающих веществ).
"""
from pathlib import Path
from ..data import SpectraDataset, load_wide_csv

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"


def load() -> SpectraDataset:
    raise NotImplementedError("Заполнить после изучения файлов датасета")

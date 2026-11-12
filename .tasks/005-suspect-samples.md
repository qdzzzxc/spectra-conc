# Задача 005: пометка подозрительных образцов валидации и проверки в двух вариантах

## Зачем
В валидации NTNU два состава размечены как глюкоза 50 мМ, но ВСЕ модели (PLS, ridge, SVR) стабильно предсказывают ~25 мМ во всех повторах. Вероятная ошибка приготовления/метки в источнике. Метки НЕ правим, но все проверки на валидации считаем в двух вариантах: «все образцы» и «без подозрительных».

Подозрительные составы (поле `composition` в meta, формат `f"{v:.2f}"` через `|`, порядок glucose|lactate|acetaminophen|caffeine|ethanol):
- `"50.00|15.00|0.15|0.10|34.50"`
- `"50.00|15.00|0.15|0.05|69.00"`

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/loaders/ntnu.py`
- `scripts/run_ntnu_checks.py`
- `tests/test_ntnu_loader.py`

## 1. `src/spectra/loaders/ntnu.py`
- После `WORK_WINDOWS` добавить:
  ```python
  # валидация: метка 50 мМ, все модели предсказывают ~25 мМ — вероятная ошибка приготовления/метки в источнике
  SUSPECT_COMPOSITIONS: tuple[str, ...] = ("50.00|15.00|0.15|0.10|34.50", "50.00|15.00|0.15|0.05|69.00")
  ```
- В `_add_structure` добавить колонку `suspect` (bool) = `composition ∈ SUSPECT_COMPOSITIONS`.
- В `load` новый параметр (последним) `drop_suspect: bool = False`: если True — выкинуть строки с `suspect == True` ПОСЛЕ формирования meta/X (и до факторизации групп), `meta.reset_index(drop=True)`. Если что-то выкинуто — к `name` добавить суффикс `"_nosus"`. Docstring `load` — одна строка про `drop_suspect`.

## 2. `scripts/run_ntnu_checks.py`
- Загрузить ещё `val_clean = load("validation", windows="work", drop_suspect=True)` и напечатать его `summary()`.
- Шаги «внешняя валидация», «исключение интерферента», «кривая обучения» выполнять ДВАЖДЫ: с `val` и с `val_clean`. В каждую строку результата добавить колонку `val_set` со значением `"all"` или `"clean"` (первой колонкой). Результаты обоих вариантов — в те же CSV (concat).
  - `{out}_external_pred.csv` — тоже с колонкой `val_set`.
- Leave-one-day-out не меняется (он только на калибровке).
- Графики:
  - `ntnu_learning_curve.png`: линии для обоих вариантов; `"all"` — пунктир (`ls="--"`), `"clean"` — сплошная; легенда `prep | model (all)` / `prep | model (clean)`.
  - `ntnu_interferents.png`: строить по `val_clean`, а подозрительные точки из `val` добавить серыми крестиками (`marker="x", color="gray"`, подпись «подозрительные (исключены)») — только для варианта `"full"`. Наклоны в легенде — по `val_clean`. В заголовок фигуры добавить «(валидация без подозрительных образцов)».
- Печать итоговых таблиц — как раньше, теперь с колонкой `val_set`.

## 3. `tests/test_ntnu_loader.py` — добавить тесты (существующие не трогать)
1. `load("validation")`: `meta["suspect"].sum() == 6`; `set(meta.loc[meta.suspect, "glucose"]) == {50.0}`.
2. `load("validation", drop_suspect=True)`: `len == 114`, `"suspect"` в meta и `not meta.suspect.any()`, имя оканчивается на `"_nosus"`, `X.shape[0] == 114`.
3. `load("calibration", drop_suspect=True)`: `len == 273` (в калибровке подозрительных нет), имя без `"_nosus"`.

## Приёмка (выполни сам)
```bash
uv run pytest -q
uv run python scripts/run_ntnu_checks.py --quick --out /tmp/claude-1000/qwen005/ntnu_checks
```
Ожидание: все тесты зелёные; в таблице внешней валидации строка `clean` для `pls snv` с RMSE около 1.1. Отчёт: изменённые файлы, вывод pytest, ПОЛНЫЕ итоговые таблицы скрипта.

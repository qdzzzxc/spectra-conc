# Задача 004: специальные проверки устойчивости (интерференты, другой день, кривая обучения)

## Контекст
Прочитай: `CLAUDE.md`, `src/spectra/data.py`, `src/spectra/evaluate.py`, `src/spectra/models.py`, `src/spectra/metrics.py`, `src/spectra/loaders/ntnu.py`.
Обозначения: `cal = load("calibration", windows="work")`, `val = load("validation", windows="work")` из `spectra.loaders.ntnu`. Валидация — отдельно смешанные растворы, отдельный день; общих составов с калибровкой нет.
«Пайплайн» = пара `(model, prep)`, например `("pls", "snv")`.

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/checks.py` (новый)
- `scripts/run_ntnu_checks.py` (новый)
- `tests/test_checks.py` (новый)

## 1. `src/spectra/checks.py`
Docstring модуля: «Специальные проверки устойчивости: исключение интерферента, перенос на другой день, внешняя валидация, кривая обучения.»
Импорты: `numpy`, `pandas`, `scipy.stats.linregress`, из `.data` — `SpectraDataset`, из `.evaluate` — `_fit_with_inner_cv`, из `.metrics` — `regression_metrics`, из `.models` — `build`.
Тип: `Pipe = tuple[str, str]`.

Функции (аннотации, docstring на русском):

1. `subset(ds: SpectraDataset, mask: np.ndarray) -> SpectraDataset` — новый датасет из строк `mask` (bool-массив): `X[mask]`, тот же `wl`, `y[mask]`, `groups[mask]`, `meta[mask].reset_index(drop=True)`, тот же `name`.

2. `fit_predict(train: SpectraDataset, X_test: np.ndarray, pipe: Pipe, seed: int = 0) -> np.ndarray` — `build(model, prep, train.X.shape[1], n_train=len(train))`, затем `_fit_with_inner_cv(p, grid, train.X, train.y, train.groups, seed=seed)`, вернуть `predict(X_test)` как 1D float-массив.

3. `_metrics_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]` — `regression_metrics(y, pred)`, оставить ключи `rmse, mae, bias, r2, n`.

4. `external_validation(cal, val, pipes: list[Pipe], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]`
   Для каждого pipe: обучить на всей `cal`, предсказать `val`.
   Возврат: (таблица: `model, prep, n_train, n_train_groups, rmse, mae, bias, r2, n`; предсказания: `model, prep, idx, y_true, y_pred` для каждого спектра `val`).

5. `interferent_exclusion(cal, val, interferent: str, pipes: list[Pipe], seed: int = 0, zero_tol: float = 0.1) -> pd.DataFrame`
   Для каждого pipe и для двух вариантов обучения:
   - `"without"`: обучение на строках `cal`, где `cal.meta[interferent] < zero_tol`;
   - `"full"`: обучение на всей `cal`.
   Тест — вся `val`. Остаток `res = pred - val.y`. `lr = linregress(val.meta[interferent], res)`.
   Строка: `interferent, train_set, model, prep, n_train, n_train_groups, rmse, mae, bias, r2, n, slope=lr.slope, slope_se=lr.stderr, slope_p=lr.pvalue`.
   (slope — «дозовый отклик»: на сколько мМ сдвигается предсказание глюкозы на 1 мМ интерферента.)

6. `leave_one_day_out(cal, pipes: list[Pipe], seed: int = 0) -> pd.DataFrame`
   Для каждого `day` из `sorted(cal.meta["day"].unique())`:
   - test = строки этого дня;
   - train = строки других дней, ИСКЛЮЧАЯ группы, которые встречаются в test (`~np.isin(cal.groups, test_groups)`); — иначе один и тот же состав окажется по обе стороны;
   - если в train меньше 5 групп — пропустить день.
   Строка: `day, model, prep, n_train, n_train_groups, n_dropped_groups, rmse, mae, bias, r2, n`, где `n_dropped_groups` — число групп из других дней, выкинутых из-за пересечения.

7. `learning_curve(cal, val, pipes: list[Pipe], sizes: list[int], seeds: list[int]) -> pd.DataFrame`
   Уникальные группы `cal`: `ug = np.unique(cal.groups)`. Для каждого `size` (если `size > len(ug)` — взять `len(ug)`), каждого `seed`, каждого pipe:
   `rng = np.random.default_rng(seed)`; `chosen = rng.choice(ug, size=size, replace=False)`; train = `subset(cal, np.isin(cal.groups, chosen))`; обучить (`fit_predict(..., seed=seed)`), предсказать `val`.
   Строка: `n_groups=size, seed, model, prep, n_train, rmse, mae, bias, r2, n`.
   Выбор групп зависит только от `(size, seed)` — одинаковый для всех pipe.

## 2. `scripts/run_ntnu_checks.py`
Стиль как `scripts/run_experiment.py` (sys.path.insert, argparse). Аргументы:
- `--out` (default `reports/ntnu_checks`) — префикс выходных файлов;
- `--pipes` (nargs="*", default `["pls:snv", "pls:sg1", "ridge:snv"]`) — формат `model:prep`;
- `--quick` — если `--pipes` не задан явно: `["pls:snv"]`; `sizes=[10, 30, 56]`, `seeds=[0, 1]`; иначе `sizes=[6, 10, 15, 20, 30, 40, 56]`, `seeds=[0, 1, 2, 3, 4]`;
- `--seed` (int, default 0) — для остальных проверок.

Шаги `main()` (на каждом шаге печать в stdout, имя шага и время):
1. `cal`, `val` с `windows="work"`; печать `summary()` обоих.
2. `external_validation` → `{out}_external.csv`, `{out}_external_pred.csv`.
3. `interferent_exclusion` для `"lactate"` и `"ethanol"` → один CSV `{out}_interferents.csv`.
4. `leave_one_day_out` → `{out}_lodo.csv`.
5. `learning_curve` → `{out}_learning.csv`.
6. Графики (matplotlib Agg, dpi=120, подписи на русском с единицами) в `reports/figures/`:
   - `ntnu_learning_curve.png`: x = число групп (составов) в обучении, y = RMSEP на валидации, мМ; для каждого pipe — линия среднего по seed и полоса ±std (`fill_between`, alpha=0.2); легенда `prep | model`.
   - `ntnu_interferents.png`: 1×2 подграфика (лактат, этанол). Только первый pipe из списка. На каждом: точки остатка `pred − y` (мМ) против концентрации интерферента (мМ) на валидации, для `"without"` и `"full"` разными цветами; линии регрессии; в легенде slope с 2 знаками («без интерферента: 0.46 мМ/мМ»). Для этого `interferent_exclusion` недостаточно — пересчитать предсказания в скрипте через `fit_predict` (это допустимо, функция дешёвая).
   - `ntnu_lodo.png`: столбцы RMSE по дням (x = «день N»), группировка по pipe.
7. Печать итоговых таблиц (`to_string(index=False)`, округление до 3 знаков).

## 3. `tests/test_checks.py`
Стиль импорта как `tests/test_smoke.py`. Данные — синтетика:
```python
def _pair():
    cal = make_synthetic(n_solutions=20, seed=10)
    val = make_synthetic(n_solutions=10, seed=11)
    cal.meta["day"] = np.where(cal.groups < 10, 1, 2)
    return cal, val
```
(в `make_synthetic` мета-колонка интерферента называется `"interferent"`.)
Тесты:
1. `subset` сохраняет согласованность: `len`, `X.shape[0]`, `len(meta)` равны числу True в маске.
2. `external_validation(cal, val, [("pls","raw")])`: в таблице 1 строка, `r2 > 0.9`; в предсказаниях `len == len(val)`.
3. `interferent_exclusion(cal, val, "interferent", [("pls","raw")], zero_tol=30)`: 2 строки (`without`, `full`), колонки `slope`, `slope_se` конечные; `n_train` у `without` < у `full`.
4. `leave_one_day_out(cal, [("pls","raw")])`: 2 строки (дни 1 и 2), `n_dropped_groups == 0` (группы по дням не пересекаются).
5. `learning_curve(cal, val, [("pls","raw")], sizes=[5, 50], seeds=[0, 1])`: 4 строки; `n_groups` ⊆ {5, 20} (50 обрезано до 20).

## Приёмка (выполни сам)
```bash
uv run pytest -q
uv run python scripts/run_ntnu_checks.py --quick --out /tmp/claude-1000/qwen004/ntnu_checks
```
Ожидание: все тесты зелёные; скрипт отрабатывает, печатает таблицы, создаёт 3 PNG в `reports/figures/` и 5 CSV по префиксу. Отчёт: изменённые файлы, вывод pytest, итоговые таблицы скрипта (полностью).

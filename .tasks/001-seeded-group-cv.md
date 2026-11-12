# Задача 001: seed в групповой CV, разброс по фолдам, размер сетки PLS, флаги CLI

## Контекст
Проект: регрессия концентрации по спектрам. Пакет `src/spectra/`. Окружение: uv (НЕ pip).
Прочитай `CLAUDE.md` (жёсткие правила) и файлы, перечисленные ниже, ПЕРЕД изменениями.

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/evaluate.py`
- `src/spectra/models.py`
- `scripts/run_experiment.py`
- `src/spectra/loaders/synthetic.py` (новый)
- `tests/test_evaluate.py` (новый)

Любые другие файлы НЕ трогать. Не делать git commit, не менять pyproject.toml/uv.lock, не ставить пакеты.

## Изменения

### 1. `src/spectra/evaluate.py`
1.1. Добавить публичную функцию:
```python
def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Внешние фолды по группам: GroupKFold с перемешиванием групп, детерминированно по seed."""
```
Реализация: `n = min(n_splits, len(np.unique(groups)))`, `GroupKFold(n_splits=n, shuffle=True, random_state=seed)`, вернуть `list(cv.split(np.zeros(len(groups)), groups=groups))`.

1.2. `_fit_with_inner_cv(pipe, grid, X, y, groups, inner_splits=4, seed=0)`: внутренний `GroupKFold` тоже с `shuffle=True, random_state=seed`. Остальное без изменений.

1.3. `nested_group_cv(ds, model, prep, outer_splits=5, seed=0) -> tuple[np.ndarray, np.ndarray]`:
- фолды брать из `group_folds(ds.groups, outer_splits, seed)`;
- `build(...)` вызывать ВНУТРИ цикла по фолдам с `n_train=len(tr)` (см. п. 2);
- вернуть `(oof, fold)`, где `fold` — массив int длины `len(ds)`, номер внешнего фолда, в котором образец был в тесте.

1.4. `extrapolation_split(ds, model, prep, top_frac=0.2, seed=0)`: передавать `seed` в `_fit_with_inner_cv`, `build(...)` вызывать с `n_train=int(tr.sum())`.

1.5. `run_grid(ds, models, preps, outer_splits=5, do_extrapolation=True, verbose=True, seed=0)`:
- вызывать `nested_group_cv(..., seed=seed)` и `extrapolation_split(..., seed=seed)`;
- в строку `group_cv` добавить поля: `"seed": seed`, `"rmse_fold_mean"`, `"rmse_fold_std"` — среднее и std (ddof=1, если фолдов > 1, иначе nan) RMSE, посчитанного отдельно на каждом внешнем фолде через `regression_metrics(ds.y[f], oof[f])["rmse"]`;
- в строку `extrapolation` добавить `"seed": seed`;
- в каждую строку `oof_rows` добавить `"fold": int(fold[i])` и `"seed": seed`;
- verbose-печать: добавить `±{rmse_fold_std:.3f}` после RMSE.

### 2. `src/spectra/models.py`
- Заменить `_pls_max(n_features, n_train_hint=30)` на:
```python
def _max_components(n_features: int, n_train: int, inner_splits: int = 4) -> int:
    """Верхняя граница числа компонент PLS/PCA: не больше половины размера внутреннего train-фолда и не больше 15."""
    n_inner = n_train - n_train // inner_splits
    return int(min(15, n_features, max(1, n_inner // 2)))
```
- Сигнатура: `build(model: str, prep: str, n_features: int, n_train: int)`. Везде, где был `_pls_max(n_features)`, использовать `_max_components(n_features, n_train)`.
- Для `svr` и `gpr` число компонент PCA: `min(20, n_features, n_train - 1)` и `min(10, n_features, n_train - 1)` соответственно.

### 3. `src/spectra/loaders/synthetic.py` (новый)
```python
"""Загрузчик синтетики для проверки CLI без реальных данных."""
from ..data import SpectraDataset
from ..synthetic import make_synthetic


def load() -> SpectraDataset:
    """Синтетический датасет с параметрами по умолчанию."""
    return make_synthetic()
```

### 4. `scripts/run_experiment.py`
Добавить аргументы:
- `--quick` (store_true): если задан и `--models`/`--preps` не переданы явно, использовать `models=["pls", "ridge", "gbm"]`, `preps=["raw", "snv", "sg1"]`. Для этого сделать default у `--models` и `--preps` равным `None` и подставлять полный список (`MODELS`/`PREPROCESSORS`) или quick-список после парсинга.
- `--seed` (int, default 0) → `run_grid(..., seed=a.seed)`.
- `--outer-splits` (int, default 5) → `run_grid(..., outer_splits=a.outer_splits)`.

### 5. `tests/test_evaluate.py` (новый)
Стиль импорта как в `tests/test_smoke.py` (sys.path.insert). Тесты:
1. `test_group_folds_no_leak`: `ds = make_synthetic(n_solutions=12, seed=3)`; для каждого `(tr, te)` из `group_folds(ds.groups, 5, seed=0)`: `set(ds.groups[tr]) & set(ds.groups[te]) == set()`; объединение всех `te` покрывает все индексы ровно один раз.
2. `test_group_folds_seed`: одинаковый seed → одинаковые фолды (сравнить массивы te); seed 0 и seed 1 → хотя бы один фолд отличается.
3. `test_max_components_small`: `_max_components(400, 10) >= 1`, `_max_components(400, 10) <= 4`, `_max_components(400, 1000) == 15`, `_max_components(3, 1000) == 3`.
4. `test_run_grid_fold_stats`: `make_synthetic(n_solutions=15, seed=4)`, `run_grid(ds, ["pls"], ["raw"], outer_splits=3, do_extrapolation=False, verbose=False, seed=7)`; в строке group_cv есть `rmse_fold_std` (конечное число ≥ 0) и `seed == 7`; в oof есть колонка `fold` со значениями ⊆ {0,1,2}.

## Стиль
Как в окружающем коде: компактно, docstring на русском, аннотации типов в новых/изменённых сигнатурах. Не переписывать нетронутые функции.

## Приёмка (выполни сам и убедись, что всё проходит)
```bash
uv run pytest -q
uv run python scripts/run_experiment.py --help
uv run python scripts/run_experiment.py --loader synthetic --out /tmp/claude-1000/qwen001/syn --quick --models pls --preps raw --seed 1 --outer-splits 3
```
Последняя команда должна напечатать строку `raw pls RMSE=... ±...` и путь к html. В конце выведи краткий отчёт: какие файлы изменены и вывод pytest.

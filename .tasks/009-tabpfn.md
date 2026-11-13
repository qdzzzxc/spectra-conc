# Задача 009: модель `tabpfn` (предобученный трансформер для малых таблиц) на PCA-признаках

## Зачем
TabPFN — трансформер, предобученный на синтетических малых табличных задачах; на наших данных он не обучается, а предсказывает «по контексту» обучающей выборки. 2262 канала для него слишком много, поэтому подаём главные компоненты спектра. Пакет `tabpfn` (группа `nn`) установлен, веса v3.5 лежат в `~/.cache/tabpfn/` (лицензия принята, некоммерческое использование). Ничего не скачивать и не ставить.

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/models.py`
- `tests/test_tabpfn.py` (новый)

## 1. `src/spectra/models.py`
В `build` добавить ветку сразу после ветки `cnn` (перед `else: raise KeyError(model)`):
```python
elif model == "tabpfn":
    from tabpfn import TabPFNRegressor  # тяжёлый импорт только при выборе tabpfn; веса — ~/.cache/tabpfn
    pipe = Pipeline([pre, ("pca", PCA()), ("m", TabPFNRegressor(device="cpu", random_state=0))])
    grid = {"pca__n_components": [c for c in (10, 30) if c < n_train // 2] or [min(10, n_train - 1)]}
```
`MODELS` НЕ менять. Больше ничего в файле не менять.

## 2. `tests/test_tabpfn.py` (новый)
Импорты как в `tests/test_smoke.py` + `pathlib`. Сразу после sys.path.insert:
```python
pytest.importorskip("tabpfn")
WEIGHTS = pathlib.Path.home() / ".cache" / "tabpfn" / "tabpfn-v3.5-20260909.safetensors"
```
Затем `from spectra.models import build`.
1. `test_build_tabpfn`: `pipe, grid = build("tabpfn", "snv", 100, n_train=200)`; `grid == {"pca__n_components": [10, 30]}`; имена шагов `pipe` — `["prep", "pca", "m"]`. Второй вызов `build("tabpfn", "snv", 100, n_train=30)` → `grid == {"pca__n_components": [10]}`.
2. `test_tabpfn_fit_predict`: декоратор `@pytest.mark.skipif(not WEIGHTS.exists(), reason="нет весов TabPFN")`. `ds = make_synthetic(n_solutions=20, seed=6)`; `pipe, _ = build("tabpfn", "snv", ds.X.shape[1], n_train=len(ds))`; `pipe.set_params(pca__n_components=10)`; `pipe.fit(ds.X, ds.y)`; `p = pipe.predict(ds.X)`: форма `(len(ds),)`, всё конечно, `regression_metrics(ds.y, p)["r2"] > 0.9`.

## Приёмка (выполни сам)
Команды ровно так (без `|`, `&&`, перенаправлений):
```bash
uv run pytest -q
uv run python scripts/run_experiment.py --loader ntnu --out /tmp/claude-1000/qwen009/ntnu --models tabpfn --preps snv --no-extrapolation --outer-splits 2 --loader-kwargs '{"windows": "work"}'
```
Ожидание: 41 тест зелёный (тест с весами может идти ~30 с); вторая команда — несколько минут, печатает строку `snv tabpfn RMSE=...` (значение заранее неизвестно). Если вторая команда идёт дольше 20 минут — остановись и сообщи. Отчёт: изменённые файлы, вывод pytest, полный вывод второй команды.

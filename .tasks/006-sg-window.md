# Задача 006: окно Савицкого–Голея как часть имени предобработки

## Зачем
Сейчас `sg1`/`sg2` жёстко используют окно 11 точек. В NTNU шаг 0,5 нм, так что это 5 нм, а полосы NIR шириной десятки нм; вторая производная с таким окном в основном усиливает шум (RMSE 5–11 мМ против ~2 у линейных). Нужно перебирать окна из CLI, не ломая существующие имена.

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/preprocess.py`
- `tests/test_preprocess.py` (новый)

## 1. `src/spectra/preprocess.py`
1. В класс `SavGol` добавить docstring: «Фильтр Савицкого–Голея по оси длин волн: window — окно в точках (нечётное), poly — степень полинома, deriv — порядок производной.»
2. Добавить в модуль (после `import`-ов) `import re` и константу:
   ```python
   # sg<deriv>[w<window>]: окно в точках; без w — 11 (как раньше)
   _SG_RE = re.compile(r"^sg([12])(?:w(\d+))?$")
   ```
3. Новая функция (перед `get_preprocessor`):
   ```python
   def _parse_sg(token: str) -> SavGol | None:
       """'sg1' -> SavGol(11, 2, 1), 'sg2w31' -> SavGol(31, 2, 2); не SG-токен -> None.
       Окно должно быть нечётным и > 2 (степень полинома), иначе ValueError."""
   ```
   - `m = _SG_RE.match(token)`; нет совпадения → `return None`.
   - `deriv = int(m.group(1))`, `window = int(m.group(2)) if m.group(2) else 11`.
   - `window % 2 == 0 or window <= 2` → `raise ValueError(f"Окно SG должно быть нечётным и больше 2: {token}")`.
   - `return SavGol(window, 2, deriv)`.
4. Переписать `get_preprocessor(name: str)`:
   - Имя разбивается по `"+"` на токены (`"snv+sg1w31"` → `["snv", "sg1w31"]`).
   - Токен → шаг: `"raw"` → `FunctionTransformer(validate=False)`, `"snv"` → `SNV()`, `"msc"` → `MSC()`, иначе `_parse_sg(token)`; если `None` → `raise KeyError(name)`.
   - Один токен → вернуть шаг как есть; несколько → `make_pipeline(*steps)`.
   - Docstring: «Предобработка по имени: токены raw/snv/msc/sg1/sg2, окно SG — суффикс wN (sg1w31), цепочка через "+" (snv+sg1w31).»
   - Словарь `table` удалить.
5. `PREPROCESSORS` НЕ менять (тот же список из 7 имён — от него зависит полная сетка).

Поведение для всех 7 старых имён должно остаться прежним (sg1 = окно 11).

## 2. `tests/test_preprocess.py` (новый)
Стиль импорта как `tests/test_smoke.py` (sys.path.insert + `from spectra...`). Данные: `ds = make_synthetic(n_solutions=10)`.
1. `get_preprocessor("sg1")` — экземпляр `SavGol` с `window == 11, poly == 2, deriv == 1`; `get_preprocessor("sg2w31")` — `window == 31, deriv == 2`.
2. `get_preprocessor("snv+sg1w21")` — `sklearn.pipeline.Pipeline` из 2 шагов; второй шаг (`.steps[1][1]`) — `SavGol` с `window == 21`; `fit(ds.X).transform(ds.X)` той же формы и конечный.
3. Совпадение со старой реализацией: `get_preprocessor("snv+sg1").fit(ds.X).transform(ds.X)` равно (`np.allclose`) `SavGol(11, 2, 1).transform(SNV().transform(ds.X))`.
4. `pytest.raises(ValueError)` для `"sg1w20"` и `"sg1w1"`; `pytest.raises(KeyError)` для `"foo"` и `"snv+bar"`.

## Приёмка (выполни сам)
```bash
uv run pytest -q
uv run python scripts/run_experiment.py --loader ntnu --out /tmp/claude-1000/qwen006/ntnu --models pls --preps sg1 sg1w31 sg2w31 --no-extrapolation --loader-kwargs '{"windows": "work"}'
```
Ожидание: все тесты зелёные (было 26, станет 30); вторая команда печатает 3 строки, `sg1 pls` с RMSE около 2.38 (как раньше), `sg2w31 pls` заметно лучше 5.36. Отчёт: изменённые файлы, вывод pytest, полный вывод второй команды.

# Задача 003: окна длин волн в загрузчике NTNU и параметры загрузчика из CLI

## Зачем
В NTNU край спектра выше 2400 нм и зона 1880–2050 нм — насыщение детектора/шум. На полном диапазоне PLS с производной ломается (R² 0.30), на окнах 1100–1880 + 2050–2400 нм — работает (R² 0.98). Нужно уметь запускать эксперимент на окнах из CLI.

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/loaders/ntnu.py`
- `scripts/run_experiment.py`
- `tests/test_ntnu_loader.py`
- `tests/test_cli.py` (новый)

## 1. `src/spectra/loaders/ntnu.py`
- Добавить константу модуля сразу после `CONC_COLS`:
  ```python
  WORK_WINDOWS: list[tuple[float, float]] = [(1100.0, 1880.0), (2050.0, 2400.0)]
  ```
  с комментарием в одну строку: `# рабочие окна: без видимой области, насыщения ~1940 нм и шумного края >2400 нм`.
- Сигнатура `load(split: str = "calibration", wl_min: float | None = None, wl_max: float | None = None, windows: list[tuple[float, float]] | str | None = None) -> SpectraDataset`.
  - `windows="work"` → использовать `WORK_WINDOWS`; любая другая строка → `ValueError`.
  - Список окон: оставить каналы, попавшие ХОТЯ БЫ в одно окно `lo <= wl <= hi` (границы включительно). Окна можно передавать списком списков (из JSON) — приводить каждое к `tuple(float, float)`.
  - `wl_min`/`wl_max` применяются вместе с окнами (логическое И).
  - Если после маски не осталось каналов → `ValueError`.
  - `name`: если окна заданы — `f"ntnu_{split}_win"`, иначе как сейчас.
- Обновить docstring `load` (одна строка про `windows`).

## 2. `scripts/run_experiment.py`
- Добавить аргумент `--loader-kwargs` (str, default `"{}"`): JSON-объект, который передаётся в `load(**kwargs)`. Пример: `--loader-kwargs '{"windows": "work"}'`.
- Невалидный JSON или не-объект → `ap.error("...")` с понятным сообщением на русском.
- Больше ничего в скрипте не менять.

## 3. `tests/test_ntnu_loader.py` — добавить тесты (существующие не трогать)
1. `load("calibration", windows="work")`: все `wl` лежат в одном из `WORK_WINDOWS`; нет каналов в (1880, 2050); `X.shape[1] == len(wl)`; `name == "ntnu_calibration_win"`.
2. `load("calibration", windows=[[1200, 1300]])`: `wl.min() == 1200.0`, `wl.max() == 1300.0`, `len(wl) == 201`.
3. `load("calibration", windows="bad")` → `ValueError`; `load("calibration", windows=[(3000, 4000)])` → `ValueError`.

## 4. `tests/test_cli.py` (новый)
Один тест: через `subprocess.run([sys.executable, "scripts/run_experiment.py", "--loader", "synthetic", "--out", str(tmp_path / "x"), "--models", "pls", "--preps", "raw", "--outer-splits", "2", "--no-extrapolation", "--loader-kwargs", "not json"], cwd=<корень проекта>, capture_output=True, text=True)` — ожидать `returncode == 2` (так завершает `ap.error`). Корень проекта: `pathlib.Path(__file__).resolve().parents[1]`.

## Приёмка (выполни сам)
```bash
uv run pytest -q
uv run python scripts/run_experiment.py --loader ntnu --out /tmp/claude-1000/qwen003/ntnu --models pls --preps snv --no-extrapolation --loader-kwargs '{"windows": "work"}'
```
Ожидание: все тесты зелёные; вторая команда печатает `ntnu_calibration_win: n=273, ...` и строку `snv pls RMSE=...` с RMSE около 2 (±0.3). Отчёт: изменённые файлы, вывод обеих команд.

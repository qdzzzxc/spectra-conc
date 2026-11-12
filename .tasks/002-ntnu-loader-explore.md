# Задача 002: загрузчик NTNU и скрипт разведочных графиков

## Контекст
Данные уже скачаны в `data/raw/ntnu/`: `CalibrationData.txt` (273 спектра), `ValidationData.txt` (120 спектров).
Сначала прочитай: `CLAUDE.md`, `src/spectra/data.py`, `src/spectra/loaders/_template.py`, `src/spectra/loaders/synthetic.py`.

Формат файлов (проверено):
- кодировка **UTF-16** (с BOM), разделитель **табуляция**, имена колонок и значения дополнены пробелами справа;
- в `CalibrationData.txt` есть лишние пустые колонки с именами `Column<N>` — их надо выбросить;
- первые 9 колонок (после strip имён): `Glucose (mM)`, `Lactate (mM)`, `Acetaminophen (mM)`, `Caffeine (mM)`, `Ethanol (mM)`, `Temperature (C)`, `Kuvette`, `Day`, `Run`;
- остальные 4200 колонок — поглощение, имена = длина волны в нм, от `400` до `2499.5` с шагом 0.5;
- в `Temperature (C)` есть пропуски (NaN) — оставить NaN, не заполнять.

Чтение: `pd.read_csv(path, sep="\t", encoding="utf-16")`, затем `df.columns = [c.strip() for c in df.columns]`, затем отбросить колонки, начинающиеся с `"Column"`.

## Можно менять ТОЛЬКО эти файлы
- `src/spectra/loaders/ntnu.py` (новый)
- `scripts/explore_ntnu.py` (новый)
- `tests/test_ntnu_loader.py` (новый)

## 1. `src/spectra/loaders/ntnu.py`

Константы модуля:
```python
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "ntnu"
FILES = {"calibration": "CalibrationData.txt", "validation": "ValidationData.txt"}
INFO_COLS = {"Glucose (mM)": "glucose", "Lactate (mM)": "lactate", "Acetaminophen (mM)": "acetaminophen",
             "Caffeine (mM)": "caffeine", "Ethanol (mM)": "ethanol", "Temperature (C)": "temperature",
             "Kuvette": "cuvette", "Day": "day", "Run": "run"}
CONC_COLS = ["glucose", "lactate", "acetaminophen", "caffeine", "ethanol"]
```

Функции (docstring на русском, аннотации типов):

1. `_read_raw(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]` — читает файл `RAW / FILES[split]`, возвращает `(info, wl, X)`: `info` — 9 колонок, переименованных по `INFO_COLS`, индекс 0..n-1; `wl` — float-массив длин волн; `X` — float-массив (n, 4200). Неизвестный `split` → `KeyError`.

2. `_add_structure(info: pd.DataFrame, split: str) -> pd.DataFrame` — возвращает НОВЫЙ DataFrame (вход не мутировать) с добавленными колонками:
   - `set` = split;
   - `composition` = строка из 5 концентраций `CONC_COLS`, каждая округлена до 2 знаков, через `"|"` (например `"10.0|0.0|0.3|0.01|0.0"`; формат числа: `f"{v:.2f}"`);
   - `block` = номер блока подряд идущих строк с одинаковыми (`composition`, `cuvette`, `day`, `run`): `(key != key.shift()).cumsum() - 1`, int, с нуля;
   - `repeat` = номер строки внутри блока, с нуля (`groupby("block").cumcount()`).

3. `load(split: str = "calibration", wl_min: float | None = None, wl_max: float | None = None) -> SpectraDataset`
   - `split` ∈ {"calibration", "validation", "all"}; "all" = калибровка, затем валидация (concat, сквозной индекс; `block` валидации сдвинуть на `max(block калибровки) + 1`, чтобы блоки не совпадали);
   - если заданы `wl_min`/`wl_max` — оставить каналы `wl_min <= wl <= wl_max`;
   - `y` = `meta["glucose"]` (мМ);
   - `groups` = `pd.factorize(meta["composition"])[0]` (по всему возвращаемому набору). ГРУППА = СОСТАВ раствора: один и тот же состав измерялся в разных днях/run/кюветах, считаем это одним образцом, чтобы не было утечки;
   - `meta` = весь info + добавленные колонки;
   - `name` = `f"ntnu_{split}"`.

В docstring модуля коротко записать: единицы (мМ), что такое группа и почему (состав повторяется в разных блоках), что validation — отдельно смешанные растворы, отдельный день (day 5).

## 2. `scripts/explore_ntnu.py`

Запуск: `uv run python scripts/explore_ntnu.py` → PNG в `reports/figures/` (создать папку) + `reports/figures/ntnu_outliers.csv`. Стиль импорта — как в `scripts/run_experiment.py` (sys.path.insert). `matplotlib.use("Agg")`. Все подписи осей на русском, с единицами. Размер фигур ~ (10, 5), `dpi=120`, `tight_layout`. Функция на каждый график, `main()` в конце.

Данные: `ds = load("all")`. Цвет по глюкозе: colormap `viridis`, `Normalize(0, 50)`, colorbar с подписью «глюкоза, мМ».

Графики:
1. `ntnu_spectra_full.png` — все спектры калибровки (`meta.set == "calibration"`), x = длина волны, нм; y = поглощение, отн. ед.; линии `lw=0.5`, цвет по глюкозе. Серыми полупрозрачными вертикальными полосами (`axvspan`, alpha=0.15) отметить зоны 1880–2050 и 2400–2500 нм, подпись в легенде «сильное поглощение воды / возможное насыщение».
2. `ntnu_spectra_diff.png` — 2 подграфика рядом: диапазоны 1100–1880 нм и 2050–2400 нм; на каждом — спектры калибровки МИНУС средний спектр калибровки, цвет по глюкозе.
3. `ntnu_repeats.png` — в диапазоне 1100–2400 нм: (а) медиана по блокам стандартного отклонения между повторами внутри блока (std по строкам блока, ddof=1, затем медиана по блокам) и (б) стандартное отклонение средних спектров блоков между блоками. Обе линии на одном графике, `yscale("log")`, легенда «внутри блока (повторы)» / «между блоками».
4. `ntnu_drift.png` — 2 подграфика друг под другом, общая ось x = порядковый номер спектра в `load("all")`: сверху — среднее поглощение в диапазоне 1100–1300 нм; снизу — температура (°C). Точки цветом по `day` (дискретная палитра `tab10`, легенда «день N»). Вертикальная пунктирная линия на границе калибровка/валидация.
5. `ntnu_pca.png` — PCA (2 компоненты) на диапазоне 1100–1880 нм после SNV (`from spectra.preprocess import SNV`), PCA обучить на калибровке, применить ко всем. Scatter PC1 vs PC2: цвет по `day` (tab10), маркер `o` — калибровка, `^` — валидация. Подписи осей с долей объяснённой дисперсии, %.
6. Выбросы → `ntnu_outliers.csv`: на тех же данных, что в п.5, PCA с 5 компонентами (fit на калибровке), невязка реконструкции `q = ||x - x_hat||^2` для каждого спектра; порог = медиана(q_калибровки) + 5 * MAD(q_калибровки) (MAD = медиана |q − медиана|). CSV с колонками: `idx, set, block, repeat, glucose, day, run, cuvette, temperature, q, is_outlier` для ВСЕХ спектров, отсортировать по `q` убыванию. Напечатать в stdout число выбросов по set и топ-10 строк.

В конце `main()` напечатать пути сохранённых файлов.

## 3. `tests/test_ntnu_loader.py`
В начале модуля: если `data/raw/ntnu/CalibrationData.txt` не существует — `pytest.skip(..., allow_module_level=True)`. Стиль импорта как в `tests/test_smoke.py`.
Тесты:
1. калибровка: `X.shape == (273, 4200)`, `wl[0] == 400.0`, `wl[-1] == 2499.5`, `np.isfinite(X).all()`, `0 <= y.min()` и `y.max() <= 50`, число уникальных групп == 56.
2. валидация: `X.shape == (120, 4200)`, групп == 35, `(meta.day == 5).all()`.
3. all: `len == 393`, групп == 91; ни одна группа не встречается одновременно в calibration и validation.
4. структура: в калибровке каждый `block` целиком лежит в одной группе (`meta.groupby("block")["composition"].nunique().max() == 1`); размеры блоков ∈ {2, 3}; `repeat` < 3.
5. `load("calibration", wl_min=1100, wl_max=2400)`: все `wl` в [1100, 2400], `X.shape[1] == len(wl)`.
6. `load("bad")` → `KeyError` (`pytest.raises`).

## Приёмка (выполни сам)
```bash
uv run pytest -q
uv run python scripts/explore_ntnu.py
ls reports/figures/
```
Ожидание: все тесты зелёные (в т.ч. старые 7), 5 PNG + 1 CSV в `reports/figures/`. Отчёт: изменённые файлы, вывод pytest, вывод explore-скрипта (число выбросов и топ-10).

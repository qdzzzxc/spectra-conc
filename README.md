# spectra-conc: концентрация вещества по оптическим спектрам

Исследовательский проект: какой метод обработки сигнала взять для стенда, который измеряет концентрацию вещества в жидкости светом. Идём от открытых датасетов похожих измерений: восстанавливаем методику, честно сравниваем «предобработка × модель», переносим выводы на свой стенд.

**Главный результат** (NIR-спектры глюкозы в буфере, датасет NTNU, 0–50 мМ): простая линейная модель (PLS или ridge на SNV) даёт ошибку **≈ 1,0–1,1 мМ (≈ 19 мг/дл)** на отдельном дне и новых растворах. Нелинейные модели (SVR, градиентный бустинг) хуже в 1,6–2,7 раза. Подробно — [`reports/SUMMARY.md`](reports/SUMMARY.md).

## С чего читать
| Что | Где |
|---|---|
| Итоги и рекомендация для стенда | [`reports/SUMMARY.md`](reports/SUMMARY.md) |
| Учебник по методам: физика, предобработка, модели, честная оценка | [`docs/METHODS.md`](docs/METHODS.md) |
| Датасеты: как собраны, что в файлах, что подозрительно | [`docs/DATASETS.md`](docs/DATASETS.md) |
| Домен в одном экране | [`docs/DOMAIN.md`](docs/DOMAIN.md) |
| План работ | [`docs/PLAN.md`](docs/PLAN.md) |
| Правила проекта (для людей и для Claude Code) | [`CLAUDE.md`](CLAUDE.md) |

## Установка и проверка
Нужен [uv](https://docs.astral.sh/uv/). Окружение фиксируется `pyproject.toml` + `uv.lock`.
```bash
uv sync
uv run pytest -q                                          # 30 тестов, ~15 с
uv run python scripts/run_synthetic_demo.py --quick       # пайплайн на синтетике, ~3 мин
```

## Данные
Сырые файлы в git не хранятся. Команды скачивания и контрольные суммы — [`data/README.md`](data/README.md). Для NTNU:
```bash
mkdir -p data/raw/ntnu && cd data/raw/ntnu
for p in "113890 00_Readme.txt" "86652 CalibrationData.txt" "86651 ValidationData.txt"; do
  set -- $p; curl -sSL "https://dataverse.no/api/access/datafile/$1?format=original" -o "$2"; done
```

## Запуск экспериментов
Все команды из корня проекта. `W='{"windows": "work"}'` — рабочие окна длин волн NTNU (1100–1880 и 2050–2400 нм).

| Скрипт | Что делает | Время (2 ядра) |
|---|---|---|
| `scripts/explore_ntnu.py` | Разведка: спектры, повторы, дрейф, PCA, выбросы → `reports/figures/` | < 1 мин |
| `scripts/run_experiment.py --loader ntnu --out reports/ntnu --loader-kwargs "$W" --models pls ridge --preps raw snv sg1w51` | Сетка «предобработка × модель», вложенная групповая CV, тест экстраполяции → CSV + HTML-дашборд | ~1 мин |
| то же с `--models pls ridge pcr_ridge svr gpr` (все предобработки) | Сетка линейных и ядерных моделей | ~10 мин |
| с `--models gbm` | Градиентный бустинг | ~15–20 мин на предобработку |
| `scripts/run_ntnu_checks.py` | Внешняя валидация, исключение интерферента, перенос на другой день, кривая обучения | ~5 мин |
| `scripts/check_mendeley_raman.py` | Проверка второго датасета (отклонён) | < 1 мин |

Полезные флаги `run_experiment.py`:
- `--models pls ridge`, `--preps snv sg1w51` — свой набор (имена моделей и предобработок — в `docs/METHODS.md`);
- `--quick` — pls/ridge/gbm × raw/snv/sg1 (на NTNU долго из-за GBM, удобнее на синтетике);
- `--seed 3` — другое разбиение на фолды (проверка устойчивости);
- `--no-extrapolation` — пропустить тест экстраполяции.

Результаты: `reports/<out>_results.csv` (метрики), `reports/<out>_oof.csv` (предсказания для каждого спектра), `reports/<out>_dashboard.html` (графики; Plotly грузится с CDN). CSV, HTML и логи в git не хранятся. Полный список команд, которыми получены цифры из записки, — в конце `reports/SUMMARY.md`.

## Структура кода
```
src/spectra/
  data.py         SpectraDataset: X (спектры), wl (длины волн), y (концентрация), groups (id раствора), meta
  preprocess.py   SNV, MSC, Савицкий–Голей; get_preprocessor("snv+sg1w31")
  models.py       PLS, ridge, PCR, SVR, GPR, GBM: build() → (Pipeline, сетка гиперпараметров)
  evaluate.py     вложенная CV по группам, тест экстраполяции, run_grid()
  checks.py       спецпроверки: интерферент, другой день, внешняя валидация, кривая обучения
  metrics.py      RMSE, MAE, bias, R², RPD
  dashboard.py    HTML-дашборд
  synthetic.py    синтетические спектры по закону Бера (отладка)
  loaders/        по файлу на датасет: ntnu.py, synthetic.py, _template.py
scripts/          точки входа (см. таблицу выше)
tests/            pytest
reports/          SUMMARY.md, figures/; CSV/HTML создаются скриптами
.tasks/           спецификации задач, которые реализовывала локальная модель (opencode + Qwen)
.claude/skills/   командные скиллы для Claude Code
```

## Новый датасет
1. Скачать в `data/raw/<name>/`, записать происхождение в `data/README.md`, описание — в `docs/DATASETS.md`.
2. Написать `src/spectra/loaders/<name>.py` по `_template.py`. Главное — **правильно задать `groups`**: повторы одного раствора — одна группа.
3. `uv run python scripts/run_experiment.py --loader <name> --out reports/<name> --quick`.

# spectra-conc: концентрация по спектрам

Каркас проекта для сравнения «предобработка × модель» на спектрах растворов.

## Быстрый старт
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python scripts/run_synthetic_demo.py --quick           # ~2 мин, откройте reports/synthetic_dashboard.html
```

## Запуск Claude Code
Откройте папку проекта в Claude Code. Он прочитает `CLAUDE.md` и выполнит план из раздела «Что делать в первую очередь». Первая просьба, например:
> Выполни шаги 1–5 из CLAUDE.md: проверь окружение, скачай датасет NTNU, опиши его в docs/DATASETS.md и прогони эксперимент.

## Реальный датасет
1. Скачать данные в `data/raw/<name>/`.
2. Написать `src/spectra/loaders/<name>.py` с функцией `load()` (шаблон `_template.py`).
3. `python scripts/run_experiment.py --loader <name> --out reports/<name>`.

Заметка: дашборд подгружает Plotly с CDN (нужен интернет при открытии).

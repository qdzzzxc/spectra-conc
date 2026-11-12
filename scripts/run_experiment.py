"""Прогон на реальном датасете. Сначала напишите загрузчик в src/spectra/loaders/<name>.py,
который возвращает SpectraDataset, потом:  python scripts/run_experiment.py --loader ntnu --out reports/ntnu"""
import argparse, importlib, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from spectra.evaluate import run_grid
from spectra.dashboard import build_dashboard
from spectra.models import MODELS
from spectra.preprocess import PREPROCESSORS

QUICK_MODELS = ["pls", "ridge", "gbm"]
QUICK_PREPS = ["raw", "snv", "sg1"]

ap = argparse.ArgumentParser()
ap.add_argument("--loader", required=True, help="модуль в spectra.loaders с функцией load()")
ap.add_argument("--out", required=True, help="префикс выходных файлов, напр. reports/ntnu")
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--preps", nargs="*", default=None)
ap.add_argument("--no-extrapolation", action="store_true")
ap.add_argument("--quick", action="store_true", help="быстрый прогон: уменьшенный список моделей/предобработок")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outer-splits", type=int, default=5)
ap.add_argument("--loader-kwargs", default="{}", help='JSON-объект с аргументами load(), напр. \'{"windows": "work"}\'')
a = ap.parse_args()
try:
    loader_kwargs = json.loads(a.loader_kwargs)
except json.JSONDecodeError as e:
    ap.error(f"--loader-kwargs: некорректный JSON ({e}). Пример: '{{\"windows\": \"work\"}}'")
if not isinstance(loader_kwargs, dict):
    ap.error("--loader-kwargs: ожидается JSON-объект вида {\"параметр\": значение}.")
models = a.models if a.models is not None else (QUICK_MODELS if a.quick else MODELS)
preps = a.preps if a.preps is not None else (QUICK_PREPS if a.quick else PREPROCESSORS)
ds = importlib.import_module(f"spectra.loaders.{a.loader}").load(**loader_kwargs)
print(ds.summary())
res, oof = run_grid(ds, models, preps, outer_splits=a.outer_splits, do_extrapolation=not a.no_extrapolation, seed=a.seed)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(f"{out}_results.csv", index=False); oof.to_csv(f"{out}_oof.csv", index=False)
print(build_dashboard(res, oof, ds, f"{out}_dashboard.html", f"{ds.name}"))

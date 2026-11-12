"""Прогон на реальном датасете. Сначала напишите загрузчик в src/spectra/loaders/<name>.py,
который возвращает SpectraDataset, потом:  python scripts/run_experiment.py --loader ntnu --out reports/ntnu"""
import argparse, importlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from spectra.evaluate import run_grid
from spectra.dashboard import build_dashboard
from spectra.models import MODELS
from spectra.preprocess import PREPROCESSORS

ap = argparse.ArgumentParser()
ap.add_argument("--loader", required=True, help="модуль в spectra.loaders с функцией load()")
ap.add_argument("--out", required=True, help="префикс выходных файлов, напр. reports/ntnu")
ap.add_argument("--models", nargs="*", default=MODELS)
ap.add_argument("--preps", nargs="*", default=PREPROCESSORS)
ap.add_argument("--no-extrapolation", action="store_true")
a = ap.parse_args()
ds = importlib.import_module(f"spectra.loaders.{a.loader}").load()
print(ds.summary())
res, oof = run_grid(ds, a.models, a.preps, do_extrapolation=not a.no_extrapolation)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(f"{out}_results.csv", index=False); oof.to_csv(f"{out}_oof.csv", index=False)
print(build_dashboard(res, oof, ds, f"{out}_dashboard.html", f"{ds.name}"))

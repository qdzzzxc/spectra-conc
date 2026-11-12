"""Отладка пайплайна на синтетике (без реальных данных).  python scripts/run_synthetic_demo.py [--quick]"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from spectra.synthetic import make_synthetic
from spectra.evaluate import run_grid
from spectra.dashboard import build_dashboard

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
ds = make_synthetic()
print(ds.summary())
models = ["pls", "ridge", "gbm"] if a.quick else ["pls", "ridge", "pcr_ridge", "svr", "gpr", "gbm"]
preps = ["raw", "snv", "sg1"] if a.quick else ["raw", "snv", "msc", "sg1", "snv+sg1"]
res, oof = run_grid(ds, models, preps)
out = pathlib.Path(__file__).resolve().parents[1] / "reports"
out.mkdir(exist_ok=True)
res.to_csv(out / "synthetic_results.csv", index=False); oof.to_csv(out / "synthetic_oof.csv", index=False)
print(build_dashboard(res, oof, ds, str(out / "synthetic_dashboard.html"), "Демо: синтетические спектры"))

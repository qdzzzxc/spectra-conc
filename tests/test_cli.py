"""Тесты CLI scripts/run_experiment.py."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_loader_kwargs_bad_json(tmp_path: pathlib.Path) -> None:
    """Невалидный JSON в --loader-kwargs -> код возврата 2 (ap.error)."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--loader",
            "synthetic",
            "--out",
            str(tmp_path / "x"),
            "--models",
            "pls",
            "--preps",
            "raw",
            "--outer-splits",
            "2",
            "--no-extrapolation",
            "--loader-kwargs",
            "not json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr

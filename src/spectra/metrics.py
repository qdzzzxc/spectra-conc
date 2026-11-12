import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y_true - y_true.mean()) ** 2))
    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rpd": sd / rmse if rmse > 0 else np.nan,   # RPD>3 - обычно "хорошо" для калибровки
        "n": int(len(y_true)),
    }

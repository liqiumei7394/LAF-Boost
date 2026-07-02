from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import DataBundle


def inverse_target(y: np.ndarray, bundle: DataBundle) -> np.ndarray:
    return y * bundle.target_std + bundle.target_mean


def metrics(y_true: np.ndarray, y_pred: np.ndarray, horizons: Sequence[int], mape_eps: float = 0.1) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, h in enumerate(horizons):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[f"mae_{h}"] = float(mean_absolute_error(yt, yp))
        out[f"rmse_{h}"] = float(mean_squared_error(yt, yp) ** 0.5)
        mask = np.abs(yt) > mape_eps
        out[f"mape_{h}"] = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else float("nan")
        out[f"r2_{h}"] = float(r2_score(yt, yp))
    out["mae_avg"] = float(np.mean([out[f"mae_{h}"] for h in horizons]))
    out["rmse_avg"] = float(np.mean([out[f"rmse_{h}"] for h in horizons]))
    return out

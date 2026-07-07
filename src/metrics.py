import numpy as np

def wmae(y_true, y_pred, is_holiday):
    """Weighted MAE, matching the Kaggle metric (holiday weeks weighted 5x)."""
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), 5.0, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))

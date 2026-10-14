"""Evaluation metrics for continuous predictions.

Three headline metrics (MAE, RMSE, R^2) plus a residual summary are kept as
separate small functions rather than one "evaluate" function, so each can be
unit tested in isolation and reused individually -- for instance, printing
just the residual summary while debugging a specific batch of predictions
without recomputing everything else.
"""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average absolute difference between true and predicted values.

    Expressed in the same units as the target (dollars, here), and less
    sensitive to a handful of very large misses than RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared difference between true and predicted values.

    Also expressed in the target's units, but penalizes large individual
    errors more heavily than MAE because errors are squared before
    averaging.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: fraction of target variance explained.

    Returns 0.0 (rather than raising or returning NaN) when y_true has zero
    variance, since "fraction of variance explained" is undefined when
    there is no variance to explain.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Summarize the distribution of prediction residuals.

    Args:
        y_true: (N,) true target values.
        y_pred: (N,) predicted values.

    Returns:
        A dict with keys 'mean' (average signed residual -- a nonzero value
        flags systematic over/under-prediction), 'std' (population standard
        deviation of the residuals, ddof=0), and 'median_abs' (median
        absolute residual, a robust alternative to MAE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median_abs": float(np.median(np.abs(residuals))),
    }

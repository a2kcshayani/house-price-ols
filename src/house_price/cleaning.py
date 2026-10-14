"""Column-wise cleaning utilities: missing-value imputation and outlier
clipping for a raw numeric feature matrix.

These are applied before any feature engineering. Every function here
returns a new array rather than mutating its input, so a raw dataset can
always be re-derived from a cleaned one for debugging.
"""

from __future__ import annotations

import warnings

import numpy as np


def impute_nan_with_mean(X: np.ndarray) -> np.ndarray:
    """Fill missing values with each column's own mean.

    Real listing data is rarely complete (a lot count might be unrecorded,
    a renovation year might be blank on an unrenovated home). Filling a gap
    with the column mean is a simple baseline that keeps every row usable
    without inventing extreme values.

    Args:
        X: (N, F) array that may contain NaN entries.

    Returns:
        (N, F) float array with no NaNs. A column that is entirely NaN is
        filled with 0.0, since a mean cannot be computed from zero
        observations.
    """
    X = np.asarray(X, dtype=float)
    out = X.copy()
    with warnings.catch_warnings():
        # An all-NaN column makes nanmean warn and return NaN; that NaN is
        # handled explicitly below, so the warning would only be noise.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        col_means = np.nanmean(X, axis=0)
    # nanmean returns NaN for an all-NaN column; treat that as "no signal".
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(out)
    if nan_mask.any():
        rows, cols = np.where(nan_mask)
        out[rows, cols] = col_means[cols]
    return out


def compute_iqr_bounds(X: np.ndarray, k: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column outlier bounds from the interquartile range.

    A handful of mansions or data-entry typos in square footage can badly
    skew a mean/std-based standardizer downstream, so bounds are derived
    from percentiles instead, which are robust to a small number of extreme
    rows.

    Args:
        X: (N, F) numeric array.
        k: IQR multiplier controlling how wide the acceptable band is.
            Larger k keeps more of the tails.

    Returns:
        (lower, upper): two (F,) arrays such that
        lower = q1 - k * iqr and upper = q3 + k * iqr, per column.
    """
    X = np.asarray(X, dtype=float)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def clip_columns(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clip every column of X into its own [lower, upper] band.

    Args:
        X: (N, F) numeric array.
        lower: (F,) per-column lower bound.
        upper: (F,) per-column upper bound.

    Returns:
        A new (N, F) array with out-of-band values pulled to the nearest
        bound. X is not modified in place.
    """
    X = np.asarray(X, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.clip(X, lower, upper)

"""Feature construction: derived ratios, categorical encoding, and
standardization for a cleaned numeric feature matrix.
"""

from __future__ import annotations

import numpy as np


def make_ratio_feature(
    numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Build a derived ratio feature with division-by-zero protection.

    Useful for engineered signals like living-area-to-lot-size ratio, where
    a handful of rows might have a recorded lot size of (near) zero.

    Args:
        numerator: (N,) array.
        denominator: (N,) array, same length as numerator.
        eps: small constant added to the denominator so the result stays
            finite even when denominator is exactly zero.

    Returns:
        (N,) array equal to numerator / (denominator + eps).
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return numerator / (denominator + eps)


def append_column(X: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Append a single feature column onto the right side of a matrix.

    Args:
        X: (N, F) matrix.
        col: (N,) column to append.

    Returns:
        A new (N, F + 1) matrix; X is not modified in place.
    """
    X = np.asarray(X, dtype=float)
    col = np.asarray(col, dtype=float).reshape(-1, 1)
    return np.hstack([X, col])


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """One-hot encode a 1-D array of categorical labels.

    Column order follows the sorted order of the distinct labels (so the
    encoding is deterministic and independent of the order labels first
    appear in the data), which matters for a categorical like a zip code
    that should be treated as a category rather than a magnitude.

    Args:
        labels: (N,) array of hashable, comparable labels (strings or ints).

    Returns:
        (N, C) float matrix where C is the number of distinct labels; row i
        has a 1.0 in the column for labels[i] and 0.0 elsewhere.
    """
    labels = np.asarray(labels)
    uniques, inverse = np.unique(labels, return_inverse=True)
    encoded = np.zeros((labels.shape[0], uniques.shape[0]), dtype=float)
    encoded[np.arange(labels.shape[0]), inverse] = 1.0
    return encoded


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and standard deviation for standardization.

    Args:
        X: (N, F) numeric array.

    Returns:
        (mean, std): two (F,) arrays. Any column with zero variance has its
        std replaced with 1.0 so later division never produces inf/NaN.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply previously fitted standardization statistics to a matrix.

    Args:
        X: (N, F) numeric array.
        mean: (F,) per-column mean, from fit_standardizer.
        std: (F,) per-column standard deviation, from fit_standardizer.

    Returns:
        A new (N, F) array equal to (X - mean) / std.
    """
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return (X - mean) / std


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so a linear model can learn an intercept.

    Args:
        X: (N, F) numeric array.

    Returns:
        A new (N, F + 1) array with column 0 equal to 1.0.
    """
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])

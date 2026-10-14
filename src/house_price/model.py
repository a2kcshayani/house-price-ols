"""Ordinary least squares regression: fitting and prediction.

Fitting is delegated to numpy.linalg.lstsq rather than forming and
inverting X.T @ X explicitly. lstsq uses a singular-value decomposition
internally, which stays numerically stable even when features are
correlated (e.g. living area and above-grade area on a house without a
basement carry nearly the same information) -- a case where explicit matrix
inversion can amplify floating point error or fail outright.
"""

from __future__ import annotations

import numpy as np


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit ordinary least squares weights, including the bias term.

    Args:
        X: (N, D) design matrix that already includes a bias column
            (see features.add_bias_column).
        y: (N,) target vector.

    Returns:
        (D,) weight vector theta minimizing ||X @ theta - y||^2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta, _residuals, _rank, _singular_values = np.linalg.lstsq(X, y, rcond=None)
    return theta


def ols_predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Score rows with a fitted linear model.

    Args:
        X: (N, D) design matrix, same column layout used to fit theta.
        theta: (D,) fitted weight vector.

    Returns:
        (N,) predicted targets.
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return X @ theta

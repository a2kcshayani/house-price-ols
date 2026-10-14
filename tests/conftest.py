"""Shared fixtures for house_price tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def linear_dataset():
    """A small, exactly-linear synthetic dataset for pipeline-level tests.

    y is an exact linear function of X's two columns plus an intercept, so a
    correctly-wired OLS pipeline should recover it almost perfectly
    (R^2 close to 1.0) regardless of which rows land in which split.
    """
    rng = np.random.default_rng(7)
    n = 200
    X = rng.uniform(low=1.0, high=50.0, size=(n, 2))
    y = 3.5 * X[:, 0] - 1.2 * X[:, 1] + 10.0
    return X, y

import numpy as np
from numpy.testing import assert_allclose

from house_price.model import ols_fit, ols_predict


class TestOlsFit:
    def test_recovers_exact_linear_relationship(self):
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.array([3.0, 5.0, 7.0])
        theta = ols_fit(X, y)
        assert_allclose(theta, [1.0, 2.0], atol=1e-8)

    def test_intercept_only_model(self):
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([2.0, 4.0, 6.0])
        theta = ols_fit(X, y)
        assert_allclose(theta, [4.0], atol=1e-8)

    def test_handles_collinear_columns_without_raising(self):
        # second column is an exact multiple of the first: X.T @ X is
        # singular, so a naive explicit-inverse implementation would raise.
        X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 8.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        theta = ols_fit(X, y)
        assert np.isfinite(theta).all()


class TestOlsPredict:
    def test_matches_manual_dot_product(self):
        X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        theta = np.array([1.0, 0.5])
        assert_allclose(ols_predict(X, theta), [2.0, 2.5, 3.0])

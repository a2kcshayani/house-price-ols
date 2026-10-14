import numpy as np
from numpy.testing import assert_allclose

from house_price.cleaning import clip_columns, compute_iqr_bounds, impute_nan_with_mean


class TestImputeNanWithMean:
    def test_fills_with_column_mean(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        result = impute_nan_with_mean(X)
        assert_allclose(result, [[1.0, 5.0], [3.0, 4.0], [2.0, 6.0]])

    def test_all_nan_column_becomes_zero(self):
        X = np.array([[np.nan], [np.nan]])
        result = impute_nan_with_mean(X)
        assert_allclose(result, [[0.0], [0.0]])

    def test_no_nans_is_unchanged(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = impute_nan_with_mean(X)
        assert_allclose(result, X)

    def test_does_not_mutate_input(self):
        X = np.array([[1.0, np.nan]])
        X_copy = X.copy()
        impute_nan_with_mean(X)
        assert_allclose(X, X_copy, equal_nan=True)


class TestComputeIqrBounds:
    def test_matches_manual_percentile_math(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        lower, upper = compute_iqr_bounds(X, k=1.5)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        assert_allclose(lower, q1 - 1.5 * iqr)
        assert_allclose(upper, q3 + 1.5 * iqr)

    def test_constant_column_has_zero_width_bounds(self):
        X = np.array([[5.0], [5.0], [5.0], [5.0]])
        lower, upper = compute_iqr_bounds(X)
        assert_allclose(lower, [5.0])
        assert_allclose(upper, [5.0])

    def test_larger_k_widens_bounds(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        lower_a, upper_a = compute_iqr_bounds(X, k=1.5)
        lower_b, upper_b = compute_iqr_bounds(X, k=3.0)
        assert lower_b[0] < lower_a[0]
        assert upper_b[0] > upper_a[0]


class TestClipColumns:
    def test_clips_out_of_band_values(self):
        X = np.array([[1.0, 10.0], [-5.0, 3.0], [8.0, 20.0]])
        lower = np.array([0.0, 2.0])
        upper = np.array([5.0, 15.0])
        result = clip_columns(X, lower, upper)
        assert_allclose(result, [[1.0, 10.0], [0.0, 3.0], [5.0, 15.0]])

    def test_does_not_mutate_input(self):
        X = np.array([[-5.0]])
        X_copy = X.copy()
        clip_columns(X, np.array([0.0]), np.array([10.0]))
        assert_allclose(X, X_copy)

import numpy as np
from numpy.testing import assert_allclose

from house_price.features import (
    add_bias_column,
    append_column,
    apply_standardizer,
    fit_standardizer,
    make_ratio_feature,
    one_hot_encode,
)


class TestMakeRatioFeature:
    def test_basic_division(self):
        result = make_ratio_feature(np.array([1.0, 2.0]), np.array([1.0, 4.0]), eps=1e-6)
        assert_allclose(result, [1.0, 0.5], atol=1e-5)

    def test_zero_denominator_stays_finite(self):
        result = make_ratio_feature(np.array([6.0]), np.array([0.0]))
        assert np.isfinite(result).all()
        assert result[0] > 1e6


class TestAppendColumn:
    def test_appends_as_last_column(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        col = np.array([10.0, 20.0])
        result = append_column(X, col)
        assert result.shape == (2, 3)
        assert_allclose(result[:, -1], col)

    def test_does_not_mutate_input(self):
        X = np.array([[1.0, 2.0]])
        X_copy = X.copy()
        append_column(X, np.array([5.0]))
        assert_allclose(X, X_copy)


class TestOneHotEncode:
    def test_string_labels_sorted_column_order(self):
        result = one_hot_encode(np.array(["b", "a", "b", "c"]))
        # sorted uniques: a, b, c -> columns 0, 1, 2
        assert_allclose(
            result,
            [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

    def test_integer_labels(self):
        result = one_hot_encode(np.array([2, 0, 1, 2]))
        assert_allclose(
            result,
            [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

    def test_single_category(self):
        result = one_hot_encode(np.array(["only", "only"]))
        assert_allclose(result, [[1.0], [1.0]])

    def test_every_row_sums_to_one(self):
        result = one_hot_encode(np.array([5, 1, 9, 1, 5]))
        assert_allclose(result.sum(axis=1), np.ones(5))


class TestFitStandardizer:
    def test_mean_and_std(self):
        X = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
        mean, std = fit_standardizer(X)
        assert_allclose(mean, [2.0, 10.0])
        assert_allclose(std[0], np.std(X[:, 0]))

    def test_zero_variance_column_gets_std_one(self):
        X = np.array([[7.0], [7.0], [7.0]])
        _mean, std = fit_standardizer(X)
        assert_allclose(std, [1.0])


class TestApplyStandardizer:
    def test_centers_and_scales(self):
        X = np.array([[1.0, 4.0], [3.0, 8.0]])
        mean = np.array([2.0, 6.0])
        std = np.array([1.0, 2.0])
        result = apply_standardizer(X, mean, std)
        assert_allclose(result, [[-1.0, -1.0], [1.0, 1.0]])


class TestAddBiasColumn:
    def test_prepends_ones(self):
        result = add_bias_column(np.array([[2.0, 3.0], [4.0, 5.0]]))
        assert_allclose(result, [[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])

    def test_single_column_input(self):
        result = add_bias_column(np.array([[7.0]]))
        assert_allclose(result, [[1.0, 7.0]])

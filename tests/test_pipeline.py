import numpy as np
import pytest

from house_price.pipeline import (
    assemble_feature_matrix,
    evaluate_predictions,
    house_price_pipeline,
    prepare_features,
    standardize_and_add_bias,
)


class TestPrepareFeatures:
    def test_imputes_then_clips(self):
        X = np.array(
            [[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0], [4.0, 5.0], [100.0, 6.0]]
        )
        result = prepare_features(X, iqr_k=1.5)
        assert np.isfinite(result).all()
        # the extreme outlier (100.0) must have been pulled down
        assert result[4, 0] < 100.0


class TestAssembleFeatureMatrix:
    def test_appends_ratio_column(self):
        X_num = np.array([[2.0, 4.0], [6.0, 3.0]])
        result = assemble_feature_matrix(X_num, 0, 1)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, 2], [0.5, 2.0], atol=1e-6)

    def test_appends_one_hot_block_when_categoricals_given(self):
        X_num = np.array([[2.0, 4.0], [6.0, 3.0]])
        result = assemble_feature_matrix(
            X_num, 0, 1, cat_labels=np.array(["x", "y"])
        )
        assert result.shape == (2, 5)


class TestStandardizeAndAddBias:
    def test_fits_only_on_train_split(self):
        splits = {
            "X_train": np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]),
            "y_train": np.array([1.0, 2.0, 3.0]),
            "X_val": np.array([[2.0, 4.0]]),
            "y_val": np.array([2.5]),
            "X_test": np.array([[6.0, 12.0]]),
            "y_test": np.array([4.0]),
        }
        std_splits, mean, _std = standardize_and_add_bias(splits)
        assert mean.tolist() == [2.0, 4.0]
        # X_val's raw row equals the train mean exactly, so after
        # standardizing and adding a bias column it should be [1, 0, 0].
        np.testing.assert_allclose(std_splits["X_val"], [[1.0, 0.0, 0.0]])
        # y values must pass through untouched.
        assert std_splits["y_train"] is splits["y_train"]

    def test_val_and_test_never_influence_the_fitted_statistics(self):
        # If val/test leaked into the fit, mean would not equal the
        # train-only mean of 5.0.
        splits = {
            "X_train": np.array([[5.0], [5.0], [5.0]]),
            "y_train": np.array([1.0, 2.0, 3.0]),
            "X_val": np.array([[500.0]]),
            "y_val": np.array([9.0]),
            "X_test": np.array([[-500.0]]),
            "y_test": np.array([-9.0]),
        }
        _std_splits, mean, _std = standardize_and_add_bias(splits)
        assert mean.tolist() == [5.0]


class TestEvaluatePredictions:
    def test_has_exact_key_set(self):
        result = evaluate_predictions(np.array([3.0, 5.0, 7.0]), np.array([2.0, 5.0, 8.0]))
        assert set(result.keys()) == {"mae", "rmse", "r2", "residual_summary"}
        assert result["mae"] == pytest.approx(0.6666666666666666)
        assert set(result["residual_summary"].keys()) == {"mean", "std", "median_abs"}


class TestHousePricePipeline:
    def test_recovers_near_perfect_fit_on_linear_data(self, linear_dataset):
        X, y = linear_dataset
        result = house_price_pipeline(X, y, train_ratio=0.6, val_ratio=0.2, seed=42)
        assert set(result.keys()) == {
            "theta",
            "y_test",
            "y_test_pred",
            "val_metrics",
            "test_metrics",
        }
        assert result["test_metrics"]["r2"] > 0.95

    def test_supports_categorical_features(self, linear_dataset):
        X, y = linear_dataset
        rng = np.random.default_rng(3)
        cat_labels = rng.choice(["north", "south", "east", "west"], size=X.shape[0])
        result = house_price_pipeline(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=1, cat_labels=cat_labels
        )
        # bias + 2 numeric + 1 ratio + 4 one-hot categories = 8 weights
        assert result["theta"].shape == (8,)

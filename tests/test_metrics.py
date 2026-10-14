import numpy as np
import pytest

from house_price.metrics import (
    mean_absolute_error,
    r_squared,
    residual_summary,
    root_mean_squared_error,
)


class TestMeanAbsoluteError:
    def test_basic_case(self):
        result = mean_absolute_error(np.array([3.0, 5.0, 2.5]), np.array([2.5, 5.0, 4.0]))
        assert result == pytest.approx(0.6666666666666666)

    def test_perfect_predictions_are_zero(self):
        assert mean_absolute_error(np.array([10.0]), np.array([10.0])) == 0.0


class TestRootMeanSquaredError:
    def test_basic_case(self):
        # errors of 0.5, 0.0, -1.5 -> squared 0.25, 0.0, 2.25 -> mean 0.8333...
        result = root_mean_squared_error(
            np.array([3.0, 5.0, 2.5]), np.array([2.5, 5.0, 4.0])
        )
        assert result == pytest.approx(np.sqrt(2.5 / 3))

    def test_perfect_predictions_are_zero(self):
        assert root_mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

    def test_penalizes_a_single_large_error_more_than_mae_does(self):
        y_true = np.array([10.0, 10.0, 10.0, 10.0])
        y_pred = np.array([10.0, 10.0, 10.0, 2.0])  # one big miss
        assert root_mean_squared_error(y_true, y_pred) > mean_absolute_error(
            y_true, y_pred
        )


class TestRSquared:
    def test_basic_case(self):
        # ss_res = 0.5, ss_tot = 8.0 -> r2 = 1 - 0.5/8.0 = 0.9375
        result = r_squared(np.array([3.0, 5.0, 7.0]), np.array([2.5, 5.0, 7.5]))
        assert result == pytest.approx(0.9375)

    def test_constant_y_true_returns_zero(self):
        result = r_squared(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
        assert result == 0.0


class TestResidualSummary:
    def test_perfect_predictions(self):
        summary = residual_summary(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert summary == {"mean": 0.0, "std": 0.0, "median_abs": 0.0}

    def test_uses_population_std(self):
        summary = residual_summary(np.array([4.0, 6.0]), np.array([2.0, 8.0]))
        assert round(summary["mean"], 4) == 0.0
        assert round(summary["std"], 4) == 2.0
        assert round(summary["median_abs"], 4) == 2.0

    def test_keys_are_exact(self):
        summary = residual_summary(np.array([1.0]), np.array([2.0]))
        assert set(summary.keys()) == {"mean", "std", "median_abs"}

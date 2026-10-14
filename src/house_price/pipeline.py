"""End-to-end orchestration: raw features and prices in, a fitted model and
held-out metrics out.

Nothing in this module does its own numeric work -- every function here is a
call graph over the smaller building blocks in cleaning.py, features.py,
splits.py, model.py, and metrics.py. Keeping the composition layer free of
numeric logic means a bug can always be isolated to one small, independently
tested function rather than hiding inside a long pipeline function.
"""

from __future__ import annotations

import numpy as np

from .cleaning import clip_columns, compute_iqr_bounds, impute_nan_with_mean
from .features import (
    add_bias_column,
    append_column,
    apply_standardizer,
    fit_standardizer,
    make_ratio_feature,
    one_hot_encode,
)
from .metrics import (
    mean_absolute_error,
    r_squared,
    residual_summary,
    root_mean_squared_error,
)
from .model import ols_fit, ols_predict
from .splits import make_shuffled_indices, partition_indices, subset_xy


def prepare_features(X: np.ndarray, iqr_k: float = 1.5) -> np.ndarray:
    """Clean a raw numeric feature matrix: impute, then clip outliers.

    Args:
        X: (N, F) raw numeric array, may contain NaNs and outliers.
        iqr_k: IQR multiplier passed through to compute_iqr_bounds.

    Returns:
        (N, F) finite array with missing values imputed and extreme values
        clipped to IQR-based bounds.
    """
    imputed = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(imputed, k=iqr_k)
    return clip_columns(imputed, lower, upper)


def assemble_feature_matrix(
    X_num: np.ndarray,
    ratio_num_idx: int,
    ratio_den_idx: int,
    cat_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Build the modeling matrix: cleaned numerics + one ratio + categoricals.

    Args:
        X_num: (N, F) cleaned numeric feature matrix.
        ratio_num_idx: column index in X_num used as the ratio numerator.
        ratio_den_idx: column index in X_num used as the ratio denominator.
        cat_labels: optional (N,) array of categorical labels (e.g. zip
            code) to one-hot encode and append. Cast non-numeric-but-coded
            categoricals (like a zip code) to strings before passing them
            in, so they are treated as categories rather than as an
            ordered magnitude.

    Returns:
        (N, F + 1) matrix, or (N, F + 1 + C) when cat_labels is provided,
        where C is the number of distinct categories.
    """
    ratio = make_ratio_feature(X_num[:, ratio_num_idx], X_num[:, ratio_den_idx])
    X = append_column(X_num, ratio)
    if cat_labels is not None:
        X = np.hstack([X, one_hot_encode(cat_labels)])
    return X


def make_train_val_test(
    X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float, seed: int
) -> dict[str, np.ndarray]:
    """Shuffle and split feature/target arrays into train/val/test subsets.

    Args:
        X: (N, F) feature matrix.
        y: (N,) target vector.
        train_ratio: fraction of rows assigned to training.
        val_ratio: fraction of rows assigned to validation. The remainder
            after train_ratio and val_ratio becomes the test split.
        seed: seed for the shuffle, so the split is reproducible.

    Returns:
        A dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
    """
    shuffled = make_shuffled_indices(y.shape[0], seed)
    train_idx, val_idx, test_idx = partition_indices(shuffled, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def standardize_and_add_bias(
    splits: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Standardize every split using train-only statistics, then add bias.

    Fitting mean/std on the training split only (never on validation or
    test) is what keeps validation and test metrics honest: if statistics
    were fit on the full dataset, information about the held-out rows would
    leak into the features the model is trained on, inflating reported
    performance.

    Args:
        splits: dict with keys X_train, y_train, X_val, y_val, X_test,
            y_test, as produced by make_train_val_test.

    Returns:
        (std_splits, mean, std): std_splits carries the same six keys, with
        every X_* matrix standardized using statistics fit on X_train and
        then given a bias column; every y_* value is passed through
        unchanged. mean and std are the (F,) statistics fit on X_train.
    """
    mean, std = fit_standardizer(splits["X_train"])
    std_splits: dict[str, np.ndarray] = dict(splits)
    for key in ("X_train", "X_val", "X_test"):
        scaled = apply_standardizer(splits[key], mean, std)
        std_splits[key] = add_bias_column(scaled)
    return std_splits, mean, std


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bundle the standard metric set for a set of predictions.

    Returns:
        A dict with keys 'mae', 'rmse', 'r2', and 'residual_summary' (itself
        a dict with 'mean', 'std', 'median_abs'), computed via the
        functions in metrics.py.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "residual_summary": residual_summary(y_true, y_pred),
    }


def house_price_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
    iqr_k: float = 1.5,
    ratio_num_idx: int = 0,
    ratio_den_idx: int = 1,
    cat_labels: np.ndarray | None = None,
) -> dict:
    """Run the full pipeline: raw features and prices in, fitted model and
    held-out metrics out.

    Stages, in order: clean (prepare_features) -> engineer
    (assemble_feature_matrix) -> split (make_train_val_test) -> standardize
    with train-only statistics and add bias (standardize_and_add_bias) ->
    fit OLS on the training split (ols_fit) -> predict on validation and
    test (ols_predict) -> score both splits (evaluate_predictions).

    Args:
        X: (N, F) raw numeric feature matrix, may contain NaNs/outliers.
        y: (N,) target vector (sale price).
        train_ratio: fraction of rows used for training.
        val_ratio: fraction of rows used for validation. The remainder
            becomes the test split.
        seed: seed controlling the train/val/test shuffle.
        iqr_k: IQR multiplier used during cleaning.
        ratio_num_idx: column of the cleaned numeric matrix used as the
            ratio-feature numerator.
        ratio_den_idx: column of the cleaned numeric matrix used as the
            ratio-feature denominator.
        cat_labels: optional (N,) array of categorical labels to one-hot
            encode and include as features.

    Returns:
        A dict with keys:
            theta: (D,) fitted OLS weights, including the bias term.
            y_test: (N_test,) true test targets.
            y_test_pred: (N_test,) predicted test targets.
            val_metrics: evaluate_predictions(...) on the validation split.
            test_metrics: evaluate_predictions(...) on the test split.
    """
    cleaned = prepare_features(X, iqr_k=iqr_k)
    features = assemble_feature_matrix(
        cleaned, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels
    )
    splits = make_train_val_test(features, y, train_ratio, val_ratio, seed)
    std_splits, _mean, _std = standardize_and_add_bias(splits)

    theta = ols_fit(std_splits["X_train"], std_splits["y_train"])

    y_val_pred = ols_predict(std_splits["X_val"], theta)
    y_test_pred = ols_predict(std_splits["X_test"], theta)

    return {
        "theta": theta,
        "y_test": std_splits["y_test"],
        "y_test_pred": y_test_pred,
        "val_metrics": evaluate_predictions(std_splits["y_val"], y_val_pred),
        "test_metrics": evaluate_predictions(std_splits["y_test"], y_test_pred),
    }

"""house_price: a from-scratch, NumPy-only linear regression toolkit for
predicting home sale prices.

No scikit-learn is used anywhere in this package. Every numeric operation
(imputation, outlier clipping, encoding, standardization, splitting, ordinary
least squares, and evaluation) is implemented directly on top of NumPy so the
full mechanics of a linear regression pipeline are visible and testable end
to end.
"""

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
from .pipeline import (
    assemble_feature_matrix,
    evaluate_predictions,
    house_price_pipeline,
    make_train_val_test,
    prepare_features,
    standardize_and_add_bias,
)
from .splits import make_shuffled_indices, partition_indices, subset_xy

__all__ = [
    "impute_nan_with_mean",
    "compute_iqr_bounds",
    "clip_columns",
    "make_ratio_feature",
    "append_column",
    "one_hot_encode",
    "fit_standardizer",
    "apply_standardizer",
    "add_bias_column",
    "make_shuffled_indices",
    "partition_indices",
    "subset_xy",
    "ols_fit",
    "ols_predict",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r_squared",
    "residual_summary",
    "prepare_features",
    "assemble_feature_matrix",
    "make_train_val_test",
    "standardize_and_add_bias",
    "evaluate_predictions",
    "house_price_pipeline",
]

__version__ = "0.1.0"

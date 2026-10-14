# house-price-ols

A from-scratch linear regression pipeline for predicting home sale prices,
built entirely on NumPy — no scikit-learn anywhere in the modeling path.
Every stage of a typical regression workflow (cleaning, feature
engineering, splitting, fitting, evaluation) is implemented as a small,
independently tested function, then composed into one end-to-end pipeline.

## Why build this instead of calling `sklearn.linear_model.LinearRegression`

Scikit-learn would get you a working model in three lines. The point here
is the opposite: to make every step of the pipeline visible and owned —
what "clean" means for a specific column, how a train/val/test split
actually gets materialized, why standardization has to be fit on the
training split only, how ordinary least squares is solved under the hood.
That legibility is also what makes the test suite meaningful: every
function is small enough to have an exact, checkable contract.

## Architecture

```
src/house_price/
├── cleaning.py    impute_nan_with_mean, compute_iqr_bounds, clip_columns
├── features.py    make_ratio_feature, append_column, one_hot_encode,
│                  fit_standardizer, apply_standardizer, add_bias_column
├── splits.py      make_shuffled_indices, partition_indices, subset_xy
├── model.py       ols_fit, ols_predict
├── metrics.py     mean_absolute_error, root_mean_squared_error,
│                  r_squared, residual_summary
└── pipeline.py    prepare_features, assemble_feature_matrix,
                   make_train_val_test, standardize_and_add_bias,
                   evaluate_predictions, house_price_pipeline
```

`pipeline.py` contains no numeric logic of its own — every function in it
is a call graph over the five modules above it. That separation is
deliberate: a bug is always traceable to one small, unit-tested function
rather than hidden inside a long procedural script.

The one design decision worth calling out explicitly: **standardization
statistics (mean/std) are fit on the training split only**, then applied
unchanged to validation and test. Fitting on the full dataset before
splitting would leak information about held-out rows into the features the
model trains on, and would make validation/test metrics look better than
the model would actually perform on genuinely new listings.

## Installation

```bash
git clone <your-repo-url>
cd house-price-ols
pip install -e ".[dev]"
```

## Quickstart (synthetic data, no download required)

```python
import numpy as np
from house_price import house_price_pipeline

rng = np.random.default_rng(0)
X = rng.uniform(1, 50, size=(200, 2))          # e.g. [sqft_living, sqft_lot]
y = 3.5 * X[:, 0] - 1.2 * X[:, 1] + 10.0        # a price-like target

result = house_price_pipeline(X, y, train_ratio=0.6, val_ratio=0.2, seed=42)
print(result["test_metrics"])
```

## Running against real data (King County house sales)

This repo's default real-data target is the King County (Seattle-area)
house sales dataset. See [`data/README.md`](data/README.md) for how to
download it (not committed here — see that file for why) and the column
mapping this pipeline expects. Once downloaded:

```bash
python scripts/run_pipeline.py --csv data/raw/kc_house_data.csv
```

## Results (King County, seed 0, 60/20/20 split)

Plain OLS with the default features in `scripts/run_pipeline.py` (7 numeric columns, a living/lot area ratio, and one-hot `zipcode`):

| Split | MAE | RMSE | R² |
|---|---|---|---|
| Validation | $106,186 | $190,917 | 0.708 |
| Test | $114,014 | $212,566 | 0.695 |

Reproduce with `python scripts/run_pipeline.py --csv data/raw/kc_house_data.csv`.

## Testing

```bash
pytest --cov=src --cov-report=term-missing
```

Each module in `src/house_price/` has a matching file in `tests/` covering
its documented contract plus edge cases (all-NaN columns, zero-variance
features, constant targets, collinear design matrices, uneven split
ratios). `tests/test_pipeline.py` additionally checks the leakage
guarantee directly: fitting `standardize_and_add_bias` on a training split
must ignore wildly different values placed in validation/test.

## Linting & type checking

```bash
ruff check .
mypy src
```

## License

MIT — see [LICENSE](LICENSE).

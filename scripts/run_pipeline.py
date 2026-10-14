#!/usr/bin/env python3
"""Run the house_price_ols pipeline against the King County dataset.

Usage:
    python scripts/run_pipeline.py [--csv data/raw/kc_house_data.csv]

Prints validation and test metrics to stdout. See data/README.md for how to
obtain the CSV this script expects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from house_price import house_price_pipeline

NUMERIC_COLUMNS = [
    "sqft_living",
    "sqft_lot",
    "bedrooms",
    "bathrooms",
    "floors",
    "grade",
    "yr_built",
]
TARGET_COLUMN = "price"
CATEGORICAL_COLUMN = "zipcode"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df[NUMERIC_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    # zipcode is a category, not a magnitude -- see data/README.md.
    cat_labels = df[CATEGORICAL_COLUMN].astype(str).to_numpy()
    return X, y, cat_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/kc_house_data.csv"),
        help="Path to the King County house sales CSV.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    if not args.csv.exists():
        print(
            f"Could not find {args.csv}. See data/README.md for how to "
            "download the King County dataset.",
            file=sys.stderr,
        )
        return 1

    X, y, cat_labels = load_dataset(args.csv)

    # ratio feature: living area relative to lot size (columns 0 and 1 of
    # NUMERIC_COLUMNS above).
    result = house_price_pipeline(
        X,
        y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        ratio_num_idx=0,
        ratio_den_idx=1,
        cat_labels=cat_labels,
    )

    print("Validation metrics:")
    print(json.dumps(result["val_metrics"], indent=2))
    print("\nTest metrics:")
    print(json.dumps(result["test_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

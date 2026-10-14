"""Reproducible train/validation/test splitting.

Splitting is separated into three small steps (shuffle indices, partition
indices, materialize subsets) rather than one monolithic function, so the
shuffle can be tested for determinism independently of the partition math,
and both can be reused for cross-validation-style splits later without
touching row data at all until the very last step.
"""

from __future__ import annotations

import numpy as np


def make_shuffled_indices(n_samples: int, seed: int) -> np.ndarray:
    """Produce a reproducible random permutation of row indices.

    A seeded, dedicated random generator is used (rather than the legacy
    global NumPy RNG) so this function's output depends only on its
    arguments, never on what other code happened to run earlier in the
    process.

    Args:
        n_samples: number of rows to permute; produces indices 0..n_samples-1.
        seed: seed controlling the permutation. The same (n_samples, seed)
            pair always yields the same order.

    Returns:
        (n_samples,) int array containing each index in 0..n_samples-1
        exactly once.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(n_samples)


def partition_indices(
    indices: np.ndarray, train_ratio: float, val_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an index array into contiguous train/validation/test slices.

    The input order is respected as-is (this function does not shuffle) so
    that shuffling and partitioning stay independently testable: pass in
    already-shuffled indices to get a random split, or pass in indices
    sorted by a date column to get a time-based split.

    Args:
        indices: 1-D array of row indices to divide.
        train_ratio: fraction of indices assigned to the train slice.
        val_ratio: fraction of indices assigned to the validation slice.
            train_ratio + val_ratio must be less than 1.0; the remainder
            becomes the test slice.

    Returns:
        (train_idx, val_idx, test_idx), a partition of indices into three
        arrays that together cover every input index exactly once.
    """
    indices = np.asarray(indices)
    n = indices.shape[0]
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def subset_xy(
    X: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select matching rows from a feature matrix and target vector.

    Args:
        X: (N, F) feature matrix.
        y: (N,) target vector aligned with X's rows.
        indices: 1-D array of row positions to select.

    Returns:
        (X_sub, y_sub), selected in the order given by indices.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    return X[indices], y[indices]

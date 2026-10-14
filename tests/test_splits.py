import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from house_price.splits import make_shuffled_indices, partition_indices, subset_xy


class TestMakeShuffledIndices:
    def test_is_a_permutation(self):
        idx = make_shuffled_indices(5, seed=0)
        assert sorted(idx.tolist()) == [0, 1, 2, 3, 4]

    def test_same_seed_is_reproducible(self):
        idx_a = make_shuffled_indices(10, seed=42)
        idx_b = make_shuffled_indices(10, seed=42)
        assert_array_equal(idx_a, idx_b)

    def test_different_seeds_usually_differ(self):
        idx_a = make_shuffled_indices(20, seed=1)
        idx_b = make_shuffled_indices(20, seed=2)
        assert not np.array_equal(idx_a, idx_b)


class TestPartitionIndices:
    def test_contiguous_prefix_split(self):
        indices = np.array([3, 1, 4, 0, 2])
        train_idx, val_idx, test_idx = partition_indices(indices, 0.6, 0.2)
        assert_array_equal(train_idx, [3, 1, 4])
        assert_array_equal(val_idx, [0])
        assert_array_equal(test_idx, [2])

    def test_partition_covers_every_index_exactly_once(self):
        indices = np.arange(17)
        train_idx, val_idx, test_idx = partition_indices(indices, 0.5, 0.3)
        combined = np.concatenate([train_idx, val_idx, test_idx])
        assert sorted(combined.tolist()) == list(range(17))

    def test_uneven_split_uses_floor_and_test_gets_remainder(self):
        indices = np.arange(10)
        train_idx, val_idx, test_idx = partition_indices(indices, 0.33, 0.33)
        assert len(train_idx) == 3
        assert len(val_idx) == 3
        assert len(test_idx) == 4  # remainder absorbs the rounding


class TestSubsetXy:
    def test_selects_matching_rows(self):
        X = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]])
        y = np.array([100.0, 200.0, 300.0, 400.0])
        X_sub, y_sub = subset_xy(X, y, np.array([2, 0]))
        assert_allclose(X_sub, [[30.0, 3.0], [10.0, 1.0]])
        assert_allclose(y_sub, [300.0, 100.0])

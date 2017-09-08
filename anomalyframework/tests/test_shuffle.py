import numpy as np
import unittest

from anomalyframework import shuffle


class TestBlockShuffle(unittest.TestCase):

    def block_shuffle_test(self):
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 1, 1])
        block_size = 2
        np.random.seed(0)
        shuffled_indices, indices_to_blocks = shuffle.block_shuffle(y, block_size, one_based=True)

        self.assertListEqual(shuffled_indices, [9, 10, 6, 7, 8, 0, 1, 2, 11, 12, 3, 4, 5])
        self.assertEqual(indices_to_blocks.tolist(), [[1, 3, 5],
                                                     [2, 4, 6]])

        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 1, 1])
        block_size = 2
        np.random.seed(1)
        shuffled_indices,  indices_to_blocks = shuffle.block_shuffle(y, block_size, one_based=False)

        self.assertListEqual(shuffled_indices, [10, 8, 9, 0, 1, 2, 11, 12, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(indices_to_blocks, np.array([
            [0, 2, 4, 6], [1, 3, 5, np.nan]]))


if __name__ == '__main__':
    unittest.main()

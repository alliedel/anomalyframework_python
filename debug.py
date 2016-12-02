from pprint import pprint
from src import shuffle
import numpy as np

y = np.array([1,1,1,2,2,2,3,3,4,5,6,1,1])
block_size = 2
shuffled_indices, indices_to_blocks = shuffle.block_shuffle(y, block_size)
print(shuffled_indices)
print(indices_to_blocks)
print('shuffled_indices: {}'.format(type(shuffled_indices)))
if type(shuffled_indices) == list:
    print('Length: {}'.format(len(shuffled_indices)))
else:
    print('Shape: {}'.format(shuffled_indices.shape))

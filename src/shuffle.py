from __future__ import absolute_import

import logging
from math import floor, ceil
import numpy as np
import pickle
import sklearn
import subprocess

from . import liblinear_utils
from . import local_pyutils

ZERO_BASED = False  # until we don't support the MATLAB version
if ZERO_BASED:
    raise NotImplementedError('ZERO_BASED not supported.')



def create_all_shuffled_files(infile, pars):
    num_shuffles = pars.algorithm.permutations.num_shuffles
    split_type = pars.algorithm.permutations.split_type
    window_stride_multiplier = pars.algorithm.permutations.window_stride_multiplier
    window_size = pars.algorithm.permutations.window_size
    shuffle_size = pars.algorithm.permutations.shuffle_size
    shuffled_train_filenames = pars.paths.files.shuffled_train_filenames
    shuffled_permutation_filenames = pars.paths.files.shuffled_permutation_filenames

    """ Take a .train file and permute it according to the shuffling parameters. """
    # Get first (non-)shuffle
    X, y = liblinear_utils.read(infile, zero_based=ZERO_BASED)
    max_index = max(y)
    randomized_indices = [idx + 1 for idx in range(max_index)]  # 1-indexing from MATLAB days
    # Save original file
    subprocess.check_call('cp {} {}'.format(infile, shuffled_train_filenames[0]))
    local_pyutils.save_array(randomized_indices, shuffled_permutation_filenames[0])

    # Shuffle the file
    logging.info('Generating shuffles')

    # TODO(allie): make train files zero-based; generate them with the python library rather than
    #  the MATLAB library.

    for shuffle_index in [idx + 1 for idx in range(num_shuffles)]:
        # shuffle the frames
        logging.info('\rShuffling file lines {}/{}'.format(shuffle_index, num_shuffles))
        create_shuffle(X, y, shuffled_train_filenames[shuffle_index],
                       shuffled_permutation_filenames,
                       split_type, window_stride_multiplier, window_size, shuffle_size)


def create_shuffle(X, y, outfile_train, outfile_permutation, split_type,
                               window_stride_multiplier, window_size, shuffle_size):
    # shuffle the frames
    randomized_indices = block_shuffle(y, shuffle_size)
    liblinear_utils.write(X[randomized_indices,:], y[randomized_indices],
                          outfile_train, zero_based=ZERO_BASED)
    # Save indices for debugging
    pickle.dump(randomized_indices, outfile_permutation)


def block_shuffle(indices, block_size):
    # TODO(allie): Handle zero-indexing here too (assuming first frame index = 1
    """
    Shuffles indices according to 'blocks' of size block_size.  All 'blocks' start with index 1.
    inputs:
        indices (array-like, shape (N,)): integers to shuffle
        block_size (int): 'chunk' size: how many consecutive-valued indices to shuffle together
    returns: (shuffled_indices, block_matrix):
        permutation (list, shape (N,)): permutation for the indices array.
            Contains all values in (0,N).
        indices_to_blocks (list, (block_size,M)) where ceil(block_size*M)=N:
            indices_to_blocks_matrix[:,2] is the list of indices assigned to (shuffled) block #3.

    Example:
         shuffle.block_shuffle([1,1,1,2,2,2,3,3,4,5,6,1,1], 2)

         shuffled_indices = [9, 10, 6, 7, 8, 0, 1, 2, 11, 12, 3, 4, 5]
         indices_to_blocks = [[1,3,5],
                              [2,4,6]]
    """
    # Figure out how many blocks we need
    max_index = max(indices)
    num_blocks = (max_index) / block_size
    
    # Assign each index to a block
    indices_to_blocks = np.reshape(np.arange(block_size * num_blocks),
                                          (block_size, num_blocks), order="F")
    indices_to_blocks += 1 if not ZERO_BASED else 0
    if num_blocks % 1 == 0:
        num_blocks = int(num_blocks)
    else:
        # Add another column to the block_matrix with (locations > max_index = NaN)
        # TODO(allie): We should probably just get rid of this extra block (and not score leftovers)
        leftover_block = range(block_size * num_blocks, max_index)
        leftover_block += [1 if not ZERO_BASED else 0]
        num_blocks = int(floor(num_blocks)) + 1
        indices_to_blocks_matrix = np.c_[indices_to_blocks, local_pyutils.nans(block_size)]
        indices_to_blocks_matrix[range(len(leftover_block)), -1] = leftover_block

    # Make list of shuffled index values
    shuffled_block_indices = np.random.permutation(num_blocks)
    shuffled_unique_indices = np.reshape(indices_to_blocks[:, shuffled_block_indices],
                                         (1, -1), order="F")
    shuffled_unique_indices = shuffled_unique_indices[~np.isnan(shuffled_unique_indices)]

    # Find locations of index values in indices.
    permutation = [index for unique_index in shuffled_unique_indices for index in
                        np.where(indices == unique_index)[0]]

    return permutation, indices_to_blocks

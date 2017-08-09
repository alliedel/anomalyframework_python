from __future__ import absolute_import

import logging
from math import ceil
import numpy as np
import subprocess

from . import liblinear_utils
import local_pyutils

ONE_BASED = 0  # until we don't support the MATLAB version
# if not ONE_BASED:
#     raise NotImplementedError('ZERO_BASED not supported.')


def create_all_shuffled_files(infile, outfiles_train, outfiles_permutation, num_shuffles,
                              shuffle_size, ignore_y_indices=True):
    """ Take a .train file and permute it according to the shuffling parameters.
    Note if ignore_y_indices=True, this will simply shuffle the lines of the files.  Otherwise,
    we have to do some more work to shuffle lines with the same number together, and to shuffle
    them in groups.
    """
    logging.info('Generating shuffles')

    if ignore_y_indices:
        assert shuffle_size == 1, NotImplementedError
        for shuffle_index in range(num_shuffles):
            logging.info('Shuffling file lines {}/{}'.format(shuffle_index+1, num_shuffles))
            subprocess.check_call(' '.join(['shuf', infile, '>', outfiles_train[shuffle_index]]),
                                  shell=True)
            subprocess.check_call(' '.join(['awk \'{print $1}\'', outfiles_train[
                shuffle_index], '>', outfiles_permutation[shuffle_index]]), shell=True)
    else:
        logging.WARNING('shuffle_size > 1 ({}).  Will take longer to shuffle.'.format(shuffle_size))

        # Get first (non-)shuffle
        X, y = liblinear_utils.read(infile, zero_based=not ONE_BASED)
        randomized_indices = [idx + ONE_BASED for idx in range(int(max(y)))]
        # Save original file
        subprocess.check_call(['cp', infile, outfiles_train[0]])
        local_pyutils.save_array(randomized_indices, outfiles_permutation[0])

        # TODO(allie): make train files zero-based; generate them with the python library rather than
        #  the MATLAB library.

        for shuffle_index in [idx for idx in range(num_shuffles)]:
            # shuffle the frames
            logging.info('Shuffling file lines {}/{}'.format(shuffle_index+1, num_shuffles))
            create_shuffle(X, y,
                           outfiles_train[shuffle_index],
                           outfiles_permutation[shuffle_index],
                           shuffle_size)


def create_shuffle(X, y, outfile_train, outfile_permutation, shuffle_size):

    # shuffle the frames
    randomized_indices, _ = block_shuffle(y, shuffle_size)
    liblinear_utils.write(X[randomized_indices,:], y[randomized_indices],
                          outfile_train, zero_based=not ONE_BASED)
    # Save indices for debugging
    local_pyutils.save_array(randomized_indices, outfile_permutation)


def block_shuffle(indices, block_size, one_based=ONE_BASED):
    # TODO(allie): Handle zero-indexing here too (currently assuming first frame index = 1)
    """
    Shuffles indices according to 'blocks' of size block_size.  All 'blocks' start with index 1.
    inputs:
        indices (array-like, shape (N,)): integers to shuffle
        block_size (int): 'chunk' size: how many consecutive-valued indices to shuffle together
        one_based: If False, assumes 0 must be the starting index.  Else, assumes 1 must be.
    returns: (shuffled_indices, block_matrix):
        permutation (list, shape (N,)): permutation for the indices array.
            Contains all values in (0,N).  permutation[indices] gives a permuted array.
        indices_to_blocks (list, (block_size,M)) where ceil(block_size*M)=N:
            indices_to_blocks_matrix[:,2] is the list of indices assigned to (shuffled) block #3.

    Example:
         shuffle.block_shuffle([1,1,1,2,2,2,3,3,4,5,6,1,1], 2)

         shuffled_indices = [9, 10, 6, 7, 8, 0, 1, 2, 11, 12, 3, 4, 5]
         indices_to_blocks = [[1,3,5],
                              [2,4,6]]
    """
    one_based = int(bool(one_based))  # Handles booleans / non-0/1's
    # Figure out how many blocks we need
    max_index = max(indices)
    num_blocks = int(ceil(float(max_index + 1 - one_based) / block_size))

    # Assign each index to a block
    unique_indices = np.concatenate([
        np.arange(one_based, max_index + 1),
        local_pyutils.nans((int(block_size * num_blocks - (max_index + 1 - one_based)),))
    ])
    indices_to_blocks = np.reshape(unique_indices, (block_size, num_blocks), order="F")

    # Make list of shuffled index values
    shuffled_block_indices = np.random.permutation(num_blocks)
    shuffled_unique_indices = np.reshape(indices_to_blocks[:, shuffled_block_indices],
                                         (1, -1), order="F")
    shuffled_unique_indices = shuffled_unique_indices[~np.isnan(shuffled_unique_indices)]

    # Find locations of index values in indices.
    permutation = list([index for unique_index in shuffled_unique_indices for index in
                        np.where(indices == unique_index)[0]])
    return permutation, indices_to_blocks

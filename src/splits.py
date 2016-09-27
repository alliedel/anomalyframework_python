import numpy as np


def generate_splits(window_stride_mult, window_size, num_points, num_shuffles, shuffle_size,
                    outfile):
    """ Generates the splits per permuted set of points """

    stride = window_stride_mult * window_size
    if divmod(window_size, stride):
        raise ValueError('Window stride must be divisible by window size. Change '
                         'window_stride_mult')
    first_idx = window_size
    last_idx = num_points - 2 * window_size + 1 + window_size
    interval_start_idxs = np.arange(start=first_idx, stop=last_idx, step=stride)

    interval_end_idxs = interval_start_idxs + window_size - 1

    num_intervals = len(interval_start_idxs)

    if num_intervals < 1:
        raise ValueError('Window size should be < %d == # of frames.\n' % num_points)

    num_splits = num_intervals * (num_shuffles + 1)
    ys = np.nan(num_splits, num_points)

    for i in range(interval_start_idxs):
        ys[i, np.arange(interval_start_idxs[i], interval_end_idxs[i]+1)] = 1
        ys[i, range(interval_start_idxs[i])] = 0

    for shuffle_idx in range(num_shuffles):
        rand_idxs = block_shuffle(num_points, shuffle_size)
        for i in range(num_intervals):
            split_idx = shuffle_idx * num_intervals + i
            ys[split_idx, rand_idxs[np.arange(start=interval_start_idxs[i],
                                              stop=interval_end_idxs[i])]]



def block_shuffle(num_points, shuffle_size):
    pass
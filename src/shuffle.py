from __future__ import absolute_import

import logging
import pickle
import .
import subprocess

from . import liblinear_utils

def create_shuffled_train_file(train_file_in_order, split_type, n_shuffles,
                               window_stride_multiplier, window_size,
                               shuffle_indexing_filenames, shuffled_train_filenames):
    """ Take a .train file and permute it according to the shuffling parameters. """
    # Get first (non-)shuffle

    max_index = liblinear_utils.get_last_yval_from_libsvm_file(train_file_in_order)
    randomized_indices = [idx + 1 for idx in range(max_index)]  # 1-indexing from MATLAB days
    # Save indices
    pickle.dump(randomized_indices, shuffle_indexing_filenames[0])
    # Save original file
    subprocess.check_call('cp %s %s'.format(train_file_in_order, shuffled_train_filenames[0]))

    # Shuffle the file
    logging.info('Generating shuffles')


% Get rest of shuffles
fprintf('Generating shuffles\n')
[Y,X] = libsvmread(fn_libsvm);
N = max(Y);
for i = 1 + (1:pars.nShuffles)
    randIdxs = BlockShuffle(Y,pars.shuffleSize); % Shuffle the frames.  Then grab the intervals:
%     randIdxs = BlockShuffle(N,pars.shuffleSize); % Shuffle the frames.  Then grab the intervals:
    fprintf('\rShuffling file lines %d/%d',i-1,(pars.nShuffles))
    libsvmwrite(pars.paths.files.shufflenames_libsvm{i}, Y(randIdxs), sparse(X(randIdxs,:)));
    save(pars.paths.files.shuffleidxs{i},'randIdxs');
end

end

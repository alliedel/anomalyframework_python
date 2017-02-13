import glob
import numpy as np

import os
from src import run
from python import generalized_eigenvector
from src import liblinear_utils


if __name__ == '__main__':
    focus_pars = generalized_eigenvector.FocusPars(k=1)
    feature_hash = generalized_eigenvector.get_focus_params_hash_string(focus_pars)
    features_dir = '/home/allie/projects/focus/data/cache/Avenue/{}'.format(feature_hash)
    infile_features_list = sorted(glob.glob('{}/*.npy'.format(features_dir)))
    for infile_features in infile_features_list:
        assert os.path.isfile(infile_features), ValueError(infile_features +
                                                           ' doesn\'t exist')
        infile_features_libsvm = infile_features.replace('.npy', '.train')
        if not os.path.isfile(infile_features_libsvm):
            print('Creating the .train file for {}'.format(infile_features))
            X = np.load(infile_features).astype('float16')
            liblinear_utils.write(X, y=None, outfile=infile_features_libsvm, zero_based=True)

        lambd = 0.01
        # Run anomaly detection
        print('Running anomaly detection')

        a, pars = run.main(infile_features=infile_features_libsvm, n_shuffles=10, lambd=lambd)

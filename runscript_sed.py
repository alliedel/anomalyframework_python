import glob
import numpy as np
import os

from src import liblinear_utils
from src import run


if __name__ == '__main__':
    feature_files = sorted(glob.glob('/vmr103/adelgior/aladdin_features/LGW_*_CAM1.train'))
    params = {'n_shuffles': 10,
              'window_size': 20,
              'window_stride': 10,
              'shuffle_size': 10,
              'num_threads': 50}
              
    for feature_file in feature_files:
        infile_features_libsvm = feature_file
        for lambd in [0.01, 1, 10]:
            params['lambd']=lambd
            # Run anomaly detection
            print(feature_file)
            a, pars = run.main(infile_features=infile_features_libsvm, **params)

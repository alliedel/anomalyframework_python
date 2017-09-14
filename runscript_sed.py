import glob
import numpy as np
import os

from anomalyframework import liblinear_utils
from anomalyframework import run


if __name__ == '__main__':
    feature_files = sorted(glob.glob('/vmr103/adelgior/aladdin_features/LGW_*_CAM1.train'))
    params = {'n_shuffles': 10,
              'window_size': 20,
              'window_stride_multiplier': .5,
              'shuffle_size': 10,
              'num_threads': 50}
              
#    for feature_file in feature_files:
    for feature_file in feature_files[0:2]:
        infile_features_libsvm = feature_file
        for lambd in [10]:
            params['lambd']=lambd
            for max_buffer_size in [1000]:
                params['max_buffer_size'] = max_buffer_size
                for num_shuffles in [10]:
                    params['n_shuffles'] = num_shuffles
                    # Run anomaly detection
                    print(feature_file)
                    a, pars = run.main(infile_features=infile_features_libsvm, **params)

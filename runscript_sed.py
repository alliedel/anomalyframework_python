import glob
import numpy as np
import os

from src import liblinear_utils
from src import run


if __name__ == '__main__':
    feature_files = sorted(glob.glob('/vmr103/adelgior/aladdin_features/LGW_*_CAM1.train'))
    for feature_file in feature_files:
        for lambd in [0.01, 1, 10]:
            infile_features_libsvm = feature_file

            # Run anomaly detection
            print(feature_file)
            a, pars = run.main(infile_features=infile_features_libsvm, n_shuffles=10, lambd=lambd)

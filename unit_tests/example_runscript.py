import numpy as np
import os

from src import liblinear_utils
from src import run


if __name__ == '__main__':
    infile_features_libsvm = 'unit_tests/example.train'
    n_shuffles = 10
    lambd = 1000.0
    window_size = 10
    window_stride = 1
    a, pars = run.main(infile_features=infile_features_libsvm, n_shuffles=n_shuffles, lambd=lambd, window_size=window_size, window_stride=window_stride)

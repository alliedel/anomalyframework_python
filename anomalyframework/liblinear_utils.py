import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import subprocess
from scipy import sparse

DECIMAL_PRECISION = 6


def read(infile, zero_based=True):
    X, y = load_svmlight_file(infile, zero_based=zero_based)
    X = np.around(X, decimals=DECIMAL_PRECISION)
    return X, y


def write(X, y, outfile, zero_based=True):
    if y is None:
        y = np.arange(0, X.shape[0])
        y += 0 if zero_based else 1
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    dump_svmlight_file(X, y, outfile, zero_based=zero_based)


def get_last_yval_from_libsvm_file(train_file):
    last_line = subprocess.check_output(['tail', '-1', train_file])
    return int(last_line.split(' ', 1)[0])


def get_num_lines_from_libsvm_file(train_file):
    """ Find the max label in a train file"""
    return int(subprocess.check_output(['awk', 'END {print NR}', train_file]))

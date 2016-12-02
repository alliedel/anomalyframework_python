import sklearn
import subprocess


def read(infile, zero_based):
    X, y = sklearn.datasets.load_svmlight_file(infile, zero_based=zero_based)
    return X, y


def write(X, y, filename, zero_based):
    sklearn.datasets.dump_svmlight_file(X, y, filename, zero_based=zero_based)


def get_last_yval_from_libsvm_file(train_file):
    last_line = subprocess.check_output(['tail', '-1', train_file])
    return int(last_line.split(' ', 1)[0])


def get_num_lines_from_libsvm_file(train_file):
    """ Find the max label in a train file"""
    return int(subprocess.check_output(['awk', 'END {print NR}', train_file]))

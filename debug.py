from anomalyframework import shuffle
from anomalyframework import liblinear_utils
from anomalyframework import local_pyutils

local_pyutils.open_stdout_logger()

window_size = 100
num_shuffles = 10
infile = '/home/allie/Desktop/anomalyframework_python/data/input/features/Avenue/03_feaPCA.train'
X, y = liblinear_utils.read(infile, zero_based=False)

outfiles_train = ['/home/allie/Desktop/anomalyframework_python/data/tmp/03_feaPCA_%02d.train' % (
    i+1) for i in range(num_shuffles)]
outfiles_permutation = ['/home/allie/Desktop/anomalyframework_python/data/tmp/03_feaPCA_%02d.p' % (
    i+1) for i in range(num_shuffles)]

shuffle.create_all_shuffled_files(infile, outfiles_train, outfiles_permutation, num_shuffles,
                                  window_size)

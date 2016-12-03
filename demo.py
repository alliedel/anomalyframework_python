from src import filenames
from src import local_pyutils
from src import scoreanomalies_utils
from src import shuffle
from src.parameters import Pars

import os
import subprocess

local_pyutils.open_stdout_logger()

# Build project files
subprocess.check_call('cmake -Bbuild -H.', shell=True)
os.chdir('build')
try:
    subprocess.check_output('make')
    os.chdir('../')
except:
    os.chdir('../')
    raise

# Mark this directory as root
os.environ['ANOMALYROOT'] = os.path.abspath(os.path.curdir)
print(os.environ['ANOMALYROOT'])

train_file = './data/input/features/Avenue/03_feaPCA.train'

train_file = os.path.abspath(os.path.expanduser(train_file))
if not os.path.isfile(train_file):
    raise ValueError('{} does not exist.'.format(train_file))

pars = Pars(train_file)
pars.algorithm.n_shuffles = 10
pars.algorithm.window_size = 1000
filenames.fill_tags_and_paths(pars)

d = pars.paths.folders.path_to_tmp
if not os.path.isfile(d):
    os.makedirs(d)

shuffle.create_all_shuffled_files(pars.paths.files.infile_features,
                                  pars.paths.files.shufflenames_libsvm,
                                  pars.paths.files.shuffle_idxs,
                                  pars.algorithm.permutations.n_shuffles,
                                  pars.algorithm.permutations.shuffle_size)

for runinfo_fname, train_file \
        in zip(pars.paths.files.runinfo_fnames, pars.paths.files.shufflenames_libsvm):
    scoreanomalies_utils.write_execution_file(
        runinfo_fname=runinfo_fname,
        train_file=train_file,
        predict_directory=pars.paths.folders.path_to_tmp,
        solver_num=0,
        c=1 / pars.algorithm.discriminability.lambd,
        window_size=pars.algorithm.permutations.window_size,
        window_stride=pars.algorithm.permutations.window_stride,
        num_threads=pars.system.num_threads)

scoreanomalies_utils.run_and_wait_trainpredict_for_all_shuffles(
    pars.paths.files.done_files, pars.paths.files.runinfo_fnames, pars.paths.files.verbose_fnames,
    os.path.join(pars.system.anomalyframework_root, pars.system.path_to_trainpredict_relative))

# Sanity check to ensure score_anomalies did its job
summary_file = os.path.join(pars.paths.folders.path_to_tmp, 'summary.txt')
print('Summary file written to: ' + summary_file)
assert os.path.isfile(summary_file)

# Display
import matplotlib.pyplot as plt
import numpy as np
a = np.loadtxt(summary_file)
plt.figure()
plt.plot(a[:,4]/(1-a[:,4]))
plt.figure()
plt.plot(a[:,4]/10/(1-a[:,4]/10))
plt.show()

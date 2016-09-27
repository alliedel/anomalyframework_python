from src.parameter_classes import Paths, Tags, AlgorithmPars, SystemPars, Pars
from src import filenames
from src.liblinear_utils import write_execution_file, run_and_wait_trainpredict_for_all_shuffles
import os
import time

# Generate and save shuffled versions

# Calculate anomaly ratings for each shuffle

train_file = '/home/allie/Documents/anomalyframework_python/data/input/features/' + \
             'Avenue/03_feaPCA.train'


name = os.path.splitext(os.path.basename(train_file))[0]

pars = Pars()
pars.algorithm.lambd = 0.1
pars.algorithm.n_shuffles = 1

pars.paths = filenames.generate_all_paths(name, pars.algorithm,
                                          anomalyframework_root=pars.system.anomalyframework_root)

d = pars.paths.folders.path_to_tmp
if not os.path.isfile(d):
    os.makedirs(d)

write_execution_file(runinfo_fname=pars.paths.files.runinfo_fnames[0], train_file=train_file,
                     predict_directory=pars.paths.folders.path_to_tmp, solver_num=0,
                     c=1/pars.algorithm.lambd, window_size=100, window_stride=50, num_threads=8)

path_to_trainpredict = os.path.join(pars.system.anomalyframework_root,
                                    pars.system.path_to_trainpredict_relative)
print path_to_trainpredict
run_and_wait_trainpredict_for_all_shuffles(pars.paths.files.done_files,
                                           pars.paths.files.runinfo_fnames,
                                           pars.paths.files.verbose_fnames,
                                           path_to_trainpredict)
# Sanity check to ensure trainpredict did its job
summary_file = os.path.join(pars.paths.folders.path_to_tmp, 'summary.txt')
print summary_file
assert os.path.isfile(summary_file)

print "Hello world!"

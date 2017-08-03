import numpy as np
import os
import logging
import pickle
import shutil
import subprocess
import sys

from src import shuffle, scoreanomalies_utils, parameters

import local_pyutils


def main(**user_params):
    # Open logging file: stdout
    local_pyutils.open_stdout_logger()

    # Build project files
    logging.info('Building project files')
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

    # Load configuration
    pars = parameters.Pars(**user_params)

    d = pars.paths.folders.path_to_tmp
    if not os.path.isfile(d):
        os.makedirs(d)

    # Print same file
    
    
    # Shuffle files
    shuffle.create_all_shuffled_files(pars.paths.files.infile_features,
                                      pars.paths.files.shufflenames_libsvm,
                                      pars.paths.files.shuffle_idxs,
                                      pars.algorithm.permutations.n_shuffles,
                                      pars.algorithm.permutations.shuffle_size)

    window_stride = pars.algorithm.discriminability.window_size * \
                    pars.algorithm.discriminability.window_stride_multiplier
    window_size = pars.algorithm.discriminability.window_size
    for runinfo_fname, train_file, predict_directory \
            in zip(pars.paths.files.runinfo_fnames, pars.paths.files.shufflenames_libsvm,
                   pars.paths.folders.predict_directories):
        scoreanomalies_utils.write_execution_file(
            runinfo_fname=runinfo_fname,
            train_file=train_file,
            predict_directory=predict_directory,
            solver_num=0,
            c=1 / pars.algorithm.discriminability.lambd,
            window_size=window_size,
            window_stride=window_stride,
            num_threads=pars.system.num_threads)

    scoreanomalies_utils.run_and_wait_trainpredict_for_all_shuffles(
        pars.paths.files.done_files, pars.paths.files.runinfo_fnames,
        pars.paths.files.verbose_fnames,
        os.path.join(pars.system.anomalyframework_root, pars.system.path_to_trainpredict_relative))

    a = scoreanomalies_utils.combine_summary_files([os.path.join(outdir, 'summary.txt')
                                                    for outdir in
                                                    pars.paths.folders.predict_directories])

    results_dir = pars.paths.folders.path_to_results
    local_pyutils.mkdir_p(results_dir)
    np.save(os.path.join(results_dir, 'anomaly_ratings.npy'), a)

    pickle.dump(pars, open(os.path.join(results_dir, 'pars.pickle'), 'wb'))
    # shutil.rmtree(pars.paths.folders.path_to_tmp)

#    local_pyutils.close_stdout_logger()
    return a, pars


if __name__ == '__main__':
    main(**sys.argv)

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

    window_stride = pars.algorithm.discriminability.window_size * \
                    pars.algorithm.discriminability.window_stride_multiplier
    runinfo_fname = pars.paths.files.runinfo_fname
    train_file = pars.paths.files.infile_features
    predict_directory = pars.paths.folders.path_to_tmp
    done_file = pars.paths.files.done_file
    verbose_fname = pars.paths.files.verbose_fname
    scoreanomalies_utils.write_execution_file(
        runinfo_fname=runinfo_fname,
        train_file=train_file,
        predict_directory=predict_directory,
        solver_num=0,
        c=1 / pars.algorithm.discriminability.lambd,
        window_size=pars.algorithm.discriminability.window_size,
        window_stride=window_stride,
        num_threads=pars.system.num_threads,
        max_buffer_size=pars.algorithm.discriminability.max_buffer_size,
        block_shuffle_size=pars.algorithm.permutations.shuffle_size)

    scoreanomalies_utils.run_and_wait_trainpredict(
        done_file, runinfo_fname,
        verbose_fname,
        os.path.join(pars.system.anomalyframework_root, pars.system.path_to_trainpredict_relative))

    indices, anomalousness = scoreanomalies_utils.read_meta_summary_file(os.path.join(pars.paths.folders.path_to_tmp,
                                                      'summary.txt'))

    results_dir = pars.paths.folders.path_to_results
    local_pyutils.mkdir_p(results_dir)
    np.save(os.path.join(results_dir, 'anomaly_ratings.npy'), anomalousness)
    np.save(os.path.join(results_dir, 'locations.npy'), indices)

    pickle.dump(pars, open(os.path.join(results_dir, 'pars.pickle'), 'wb'))
    # shutil.rmtree(pars.paths.folders.path_to_tmp)

#    local_pyutils.close_stdout_logger()
    return anomalousness, pars


if __name__ == '__main__':
    main(**sys.argv)

import glob
import numpy as np
import os
import logging
import pickle
import shutil
import subprocess
import sys

from src import shuffle, scoreanomalies_utils, parameters, local_pyutils


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

    # Load configuration
    pars = parameters.Pars(**user_params)
    logging.info('Feature file: {}'.format(pars.paths.files.infile_features))

    local_pyutils.mkdir_p(pars.paths.folders.path_to_tmp)
    pickle.dump(pars, open(os.path.join(pars.paths.folders.path_to_tmp, 'pars.pickle'), 'w'))

    # Shuffle files
    logging.info('Creating shuffled versions')
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
    intermediate_results_dir = os.path.join(results_dir, 'intermediates')
    local_pyutils.mkdir_p(results_dir)
    np.save(os.path.join(results_dir, 'anomaly_ratings.npy'), a)
    pickle.dump(pars, open(os.path.join(results_dir, 'pars.pickle'), 'w'))
    # Copy over some intermediates
    os.mkdir(intermediate_results_dir)
    intermediate_globs_to_save = ['*.runinfo', '*_verbose']
    for glob_to_save in intermediate_globs_to_save:
        intermediate_files_to_save = glob.glob(os.path.join(pars.paths.folders.path_to_tmp,
                                                            glob_to_save))
        for file_to_save in intermediate_files_to_save:
            shutil.copyfile(file_to_save,
                            os.path.join(intermediate_results_dir, os.path.basename(file_to_save)))
    # Remove intermediates directory
    # shutil.rmtree(pars.paths.folders.path_to_tmp)
    print('Results saved to {}'.format(results_dir))
    return a, pars


if __name__ == '__main__':
    main(**sys.argv)

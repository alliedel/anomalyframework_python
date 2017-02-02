import os

from src import shuffle
from src import scoreanomalies_utils
from src import parameters


def main(**user_params):
    pars = parameters.Pars(**user_params)

    d = pars.paths.folders.path_to_tmp
    if not os.path.isfile(d):
        os.makedirs(d)
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

    return a, pars

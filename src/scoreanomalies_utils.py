import os
import subprocess
import time


def write_execution_file(runinfo_fname, train_file, predict_directory, solver_num, c,
                         window_size, window_stride, num_threads):
    """ Writes the execution file that trainpredict reads.  One file per shuffle. """

    file_handle = open(runinfo_fname, 'w')
    B = 1   # 'bias' term; just means we don't assume zero-centering of classifier

    str_train = "-s %d -c %d -B %f" % (solver_num, c, B)

    file_handle.write('commandLine=%s\n' % str_train)
    file_handle.write('inputFile=%s\n' % train_file)
    file_handle.write('outputDirectory=%s\n' % predict_directory)

    if window_size:
        file_handle.write('windowSize=%d\n' % window_size)

    if window_stride:
        file_handle.write('windowStride=%d\n' % window_stride)

    if num_threads:
        file_handle.write('numThreads=%d\n' % num_threads)

    file_handle.close()


def run_and_wait_trainpredict_for_all_shuffles(done_files, runinfo_fnames, verbose_fnames,
                                               path_to_trainpredict):
    """ Calls trainpredict for each shuffle and ensures the jobs finish. Wraps
    start_trainpredict_for_one_shuffle """

    assert(len(done_files) == len(runinfo_fnames) == len(verbose_fnames))
    n_shuffles = len(done_files)

    for s_idx in range(n_shuffles):
        start_trainpredict_for_one_shuffle(done_files[s_idx], path_to_trainpredict, runinfo_fnames[
            s_idx], verbose_fnames[s_idx])

    # Wait for all of the per-shuffle scoring to finish by checking for the done_file
    for s_idx in range(n_shuffles):
        done_file = done_files[s_idx]
        while not os.path.isfile(done_file):
            time.sleep(1)


def start_trainpredict_for_one_shuffle(done_file, path_to_trainpredict, runinfo_fname,
                                       verbose_fname):
    """ Starts an instance of trainpredict (for one shuffle of the data). Typically wrapped by
    run_and_wait_trainpredict_for_all_shuffles """

    cmd = "rm -f %s; %s %s >> %s; echo Done! >> %s &" % \
          (done_file, path_to_trainpredict, runinfo_fname, verbose_fname, done_file)

    subprocess.check_call(cmd, shell=True)

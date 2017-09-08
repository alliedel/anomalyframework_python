import os
import numpy as np
import subprocess

ONE_BASED = 0


def write_execution_file(runinfo_fname, train_file, predict_directory, solver_num, c,
                         window_size, window_stride, num_threads, num_shuffles, max_buffer_size,
                         block_shuffle_size):
    """ Writes the execution file that trainpredict reads.  One file per shuffle. """

    file_handle = open(runinfo_fname, 'w')
    B = 1   # 'bias' term; just means we don't assume zero-centering of classifier
    str_train = "-s {} -c {} -B {}".format(solver_num, c, B)

    file_handle.write('commandLine=%s\n' % str_train)
    file_handle.write('inputFile=%s\n' % train_file)
    file_handle.write('outputDirectory=%s\n' % predict_directory)
    file_handle.write('numShuffles=%d\n' % num_shuffles)

    if window_size:
        file_handle.write('windowSize=%d\n' % window_size)

    if window_stride:
        file_handle.write('windowStride=%d\n' % window_stride)

    if num_threads:
        file_handle.write('numThreads=%d\n' % num_threads)

    if max_buffer_size:
        file_handle.write('maxBufferSize=%d\n' % max_buffer_size)

    if block_shuffle_size:
        file_handle.write('blockShuffleSize=%d\n' % block_shuffle_size)

    file_handle.close()
    if not os.path.isfile(runinfo_fname):
         print('Error: runinfo_fname not created.')
         import ipdb; ipdb.set_trace()

def run_and_wait_trainpredict(done_file, runinfo_fname, verbose_fname,
                                               path_to_trainpredict):
    """ Calls trainpredict for each shuffle and ensures the jobs finish. Wraps
    start_trainpredict_for_one_shuffle """

    process, cmd = start_trainpredict_for_one_shuffle(done_file, path_to_trainpredict, runinfo_fname, verbose_fname)

    # Wait for scoring to finish by checking for the done_file
    out, err = process.communicate()
    if err or not os.path.isfile(done_file):
        print('Error: ')
        print(err)
        raise Exception('Command failed: {}'.format(cmd))
    else:
        # TODO(allie): assert that the summmary files exist.
        print('Done.')


def run_and_wait_trainpredict_for_all_shuffles(done_files, runinfo_fnames, verbose_fnames,
                                               path_to_trainpredict):
    """ Calls trainpredict for each shuffle and ensures the jobs finish. Wraps
    start_trainpredict_for_one_shuffle """

    assert(len(done_files) == len(runinfo_fnames) == len(verbose_fnames))
    n_shuffles = len(done_files)
    processes = [
        start_trainpredict_for_one_shuffle(done_files[s_idx], path_to_trainpredict,
                                           runinfo_fnames[s_idx], verbose_fnames[s_idx])[0]
                 for s_idx in range(n_shuffles)]
    # Wait for all of the per-shuffle scoring to finish by checking for the done_file
    for s_idx, (process, done_file) in enumerate(zip(processes, done_files)):
        print('Waiting on job {}...'.format(s_idx))
        out, _ = process.communicate()
        err = process.poll()
        if err or not os.path.isfile(done_file):
            print('Error: ')
            print(err)
        else:
            # TODO(allie): assert that the summmary files exist.
            print('Done.'.format(s_idx))


def start_trainpredict_for_one_shuffle(done_file, path_to_trainpredict, runinfo_fname,
                                       verbose_fname):
    """ Starts an instance of trainpredict (for one shuffle of the data). Typically wrapped by
    run_and_wait_trainpredict_for_all_shuffles """

    cmd = "rm -f %s; %s %s >> %s; echo Done! >> %s" % \
          (done_file, path_to_trainpredict, runinfo_fname, verbose_fname, done_file)
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE), cmd


def combine_summary_files(list_of_summary_files):
    a = None
    num_shuffles = len(list_of_summary_files)
    for summary_file in list_of_summary_files:
        summary = np.loadtxt(summary_file)
        indices = np.squeeze(summary[:,0].astype(int)) - ONE_BASED
        anomalousness_single_shuffle = np.squeeze(summary[:,4])
        if a is None:
            a = np.zeros((max(indices+ONE_BASED),), dtype=float)
            a[indices] = 1/float(num_shuffles) * anomalousness_single_shuffle
        else:
            a[indices] += 1/float(num_shuffles) * anomalousness_single_shuffle
    return a


def read_meta_summary_file(path_to_file):
    summary = np.loadtxt(path_to_file)
    indices = np.squeeze(summary[:,0].astype(int)) - ONE_BASED
    anomalousness = np.squeeze(summary[:,4])
    return indices, anomalousness

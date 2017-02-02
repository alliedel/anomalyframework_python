import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil
import subprocess

from src import local_pyutils
from src import liblinear_utils
from src import run


if __name__ == '__main__':
    # Open logging file: stdout
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

#    infile_features = 'data/input/features/Avenue/03_feaPCA_new.train'
    for videonum in [i + 1 for i in range(21)]:
        for lambd in [0.01, 0.1, 1]:
            try:
                # infile_features = '../focus/data/cache/Avenue/{:0>2}.avi_raw._feature_patch_sz__10x10x5_.' \
                #                   'image_dtype__flo at32_.search_patch_sz__10x10x5_.frame_sz_in_blocks___.' \
                #                   'image_resz__180x320_.mt__5_.search_patch_stride__10x10x5_.pca_dim__100_.' \
                #                   'pts_per_vol__3000.train'.format(videonum)
                infile_features = '/home/allie/projects/focus/data/cache/Avenue/{:0>2}.avi_raw' \
                      '._feature_patch_sz__10x10x5_.image_dtype__float32_.search_patch_sz__10x10x5_.' \
                                  'frame_sz_in_blocks___.image_resz__180x320_.mt__5_.' \
                                  'search_patch_stride__10x10x5_.pca_dim__100_.pts_per_vol__3000.train' \
                                  ''.format(videonum)
                infile_features = os.path.abspath(os.path.expanduser(infile_features))
                if not os.path.isfile(infile_features):
                    raise ValueError('{} does not exist.'.format(infile_features))

                # Run anomaly detection
                a, pars = run.main(infile_features=infile_features, n_shuffles=10, lambd=lambd)
                results_dir = pars.paths.folders.path_to_results
                local_pyutils.mkdir_p(results_dir)
                np.save(os.path.join(results_dir, 'anomaly_ratings.npy'), a)
                pickle.dump(pars, open(os.path.join(results_dir, 'pars.pickle'), 'w'))
                shutil.rmtree(pars.paths.folders.path_to_tmp + '/*')
            except:
                pass

    # # Display
    # X, y = liblinear_utils.read(open(pars.paths.files.infile_features,'w'), False)
    # plt.figure(1)
    # plt.cla()
    # plt.plot(a/(1.0-a))
    # plt.figure(2)
    # plt.cla()
    # plt.plot(a)
    # plt.figure(3)
    # plt.cla()
    # X = X.toarray()
    # plt.imshow(X.T)
    # plt.title('X')
    # plt.show(block=True)

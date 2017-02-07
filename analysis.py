import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.ndimage

# Load the FOCUS packageimport sys
import sys
sys.path.append('/home/allie/projects/focus')  # Just to remember where this path is from!
# from focus import python.features
from python import features


IM_DIR = '/home/allie/workspace/images'


def save_fig_to_workspace(filename=None, workspace_dir=IM_DIR):
    if not filename:
        filename = '%02d.png' % plt.gcf().number
    plt.savefig(os.path.join(workspace_dir, filename))


def save_all_figs_to_workspace():
    for i in plt.get_fignums():
        plt.figure(i)
        save_fig_to_workspace()

if __name__ == '__main__':
    results_dirs = sorted(glob.glob('data/results/2017_02_06/*'))
    results_dirs = results_dirs[:-1]  # last one is corrupted from memory
    fignum = 0
    plt.close('all')
    for results_dir in reversed(results_dirs):
        try:
            anomalousness = np.load(os.path.join(results_dir, 'anomaly_ratings.npy'))
            anomalousness = abs(anomalousness - 0.5)
            anomalousness = anomalousness/(1.0-anomalousness)
            pars = pickle.load(open(os.path.join(results_dir, 'pars.pickle'), 'r'))
            is_whitened = 'whiten' in os.path.basename(pars.paths.files.infile_features)
            if is_whitened:
                feats_file = pars.paths.files.infile_features.replace('_self_whiten','')
            else:
                feats_file = pars.paths.files.infile_features
            locs3 = np.load(feats_file.replace('.train', '.npy').replace(
                'raw._','raw_locs3._'))
            feature_pars = np.load(feats_file.replace('.train',
                                                                           '.npy').replace(
                'raw._','raw_pars._'))
            coords_list = np.load(feats_file.replace('.train',
                                                                            '.npy').replace(
                'raw._','raw_grid_coords_list._'))
            rs, cs, ts = coords_list[0], coords_list[1], coords_list[2]

            # TODO(allie) : scoreanomalies doesn't handle the zero-indexing case.
            assert locs3.shape[0] - len(anomalousness) <= 1
            locs3 = locs3[1:,:]
            cs = range(16)
            anomaly_ratings3 = np.zeros((len(rs), len(cs), len(ts)))
            # import ipdb; ipdb.set_trace()
            for anomaly_index in range(locs3.shape[0]):
                anomaly_ratings3[locs3[anomaly_index,0], locs3[anomaly_index,1],
                                 locs3[anomaly_index,2]] = anomalousness[anomaly_index]
            fignum += 1
            plt.figure(fignum); plt.clf()
            smooth_size = 5
            smoothed_ratings3 = scipy.ndimage.uniform_filter(anomaly_ratings3,
                                                           smooth_size)
            smoothed_by_frame = scipy.ndimage.uniform_filter(anomaly_ratings3,
                                                           smooth_size).sum(axis=1).sum(axis=0)
            smoothed_signal = smoothed_by_frame.reshape(-1,)
            signal = smoothed_signal
            # signal = anomaly_ratings3.sum(axis=1).sum(axis=0)
            # signal = anomaly_ratings3.reshape(-1,)
            plt.fill_between(range(len(signal)), signal, facecolor='#5D8AA8', alpha=0.5)
            videonum_as_str = os.path.basename(feats_file)[0:2]
            lambd = pars.algorithm.discriminability.lambd
            plt.title('video: {}\nlambda: {}\nis_whitened: {}'.format(videonum_as_str, lambd,
                                                                   is_whitened))
            print('Saving figure to {}'.format(plt.gcf().number))
            save_fig_to_workspace()

        except Exception as exc:
            print(exc)
            print('continuing...')


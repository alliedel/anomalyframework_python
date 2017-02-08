import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.ndimage
from scipy.io import loadmat
import sklearn.metrics
from src import liblinear_utils

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
    ground_truth_dir = '/home/allie/Documents/pre_internship/ECCV2016/anomalydetect/data/input' \
                       '/groundtruth/Avenue/'
    RESULTS_DATES = ['data/results/2017_02_06/*', 'data/results/2017_02_07/*']
    results_dirs = []
    for results_date_glob in RESULTS_DATES:
        results_dirs += sorted(glob.glob(results_date_glob))

    results_dirs = results_dirs[:-1]  # last one is corrupted from memory
    fignum = 0
    plt.close('all')
    for results_dir in reversed(results_dirs):
        try:
            anomalousness = np.load(os.path.join(results_dir, 'anomaly_ratings.npy'))
        except Exception as exc:
            print('Can\'t load anomaly ratings.  Continuing...')
            print(exc)
            continue
        try:
            pars = pickle.load(open(os.path.join(results_dir, 'pars.pickle'), 'r'))
            print(pars.paths.files.infile_features)
            anomalousness = abs(anomalousness - 0.5)
            anomalousness = anomalousness / (1.0 - anomalousness)
            is_whitened = 'whiten' in os.path.basename(pars.paths.files.infile_features)
            if is_whitened:
                feats_file = pars.paths.files.infile_features[
                             :pars.paths.files.infile_features.find('._self_whiten')] + '.npy'
            else:
                feats_file = pars.paths.files.infile_features
            locs3 = np.load(feats_file.replace('.train', '.npy').replace(
                'raw._', 'raw_locs3._'))
            feature_pars = np.load(feats_file.replace('.train', '.npy').replace(
                'raw._', 'raw_pars._'))
            coords_list = np.load(feats_file.replace('.train', '.npy').replace(
                'raw._', 'raw_grid_coords_list._'))
        except Exception as exc:
            print('Couldn\'t load features')
            print(exc)
            continue
        try:
            videonum_as_str = os.path.basename(feats_file)[0:2]
            gt_file = os.path.join(ground_truth_dir, videonum_as_str + '_gt_pixel.mat')
            gt3 = np.dstack(loadmat(gt_file)['volLabel'][0])
            gt_avg_per_frame = np.mean(gt3, axis=1).mean(axis=0)
        except Exception as exc:
            print('Couldn\'t load ground truth\n')
            print(exc)
            continue
        try:
            rs, cs, ts = coords_list[0], coords_list[1], coords_list[2]

            # TODO(allie) : scoreanomalies doesn't handle the zero-indexing case.
            assert locs3.shape[0] - len(anomalousness) <= 1
            locs3 = locs3[1:, :]
            cs = range(16)
            anomaly_ratings3 = np.zeros((len(rs), len(cs), len(ts)))
            # import ipdb; ipdb.set_trace()
            for anomaly_index in range(locs3.shape[0]):
                anomaly_ratings3[locs3[anomaly_index, 0], locs3[anomaly_index, 1],
                                 locs3[anomaly_index, 2]] = anomalousness[anomaly_index]

            fignum += 1
            plt.figure(fignum); plt.clf()
            smooth_size = 5
            smoothed_ratings3 = scipy.ndimage.uniform_filter(anomaly_ratings3,
                                                             smooth_size)
            smoothed_by_frame = scipy.ndimage.uniform_filter(anomaly_ratings3,
                                                             smooth_size).sum(axis=1).sum(axis=0)
            smoothed_signal = smoothed_by_frame.reshape(-1, )
            signal = np.concatenate((smoothed_signal, np.zeros(len(gt_avg_per_frame) - len(
                smoothed_signal))))
            gt_binary_per_frame = gt_avg_per_frame > 0
            fpr, tpr, _ = sklearn.metrics.roc_curve(gt_binary_per_frame, signal)
            auc = sklearn.metrics.auc(fpr, tpr)
            corr = np.corrcoef(signal, gt_avg_per_frame)[0, 1]
            np.savez(pars.paths.folders.path_to_results, 'roc_per_frame.npz', fpr=fpr, tpr=tpr,
                     auc=auc, corr=corr)
            # Create figure
            plt.subplot(2, 1, 1)
            plt.fill_between(range(len(signal)), signal, facecolor='#5D8AA8', alpha=0.5)
            plt.subplot(2, 1, 2)
            plt.fill_between(range(len(gt_avg_per_frame)), gt_avg_per_frame)
            plt.title('Ground truth')
            lambd = pars.algorithm.discriminability.lambd
            # X,y = liblinear_utils.read(feats_file, zero_based=True)
            # d = X.shape[1]
            if 'n_components' in os.path.basename(pars.paths.files.infile_features):
                feats_basename = os.path.basename(pars.paths.files.infile_features)
                d_str = feats_basename[feats_basename.find('n_components_pca') +
                                       len('n_components_pca') + 2:feats_basename.find(
                    '.', feats_basename.find('n_components_pca'))]
            else:
                d_str = 'None'
            plt.suptitle('video: {} lambda: {}\n AUC: {} Corr:{}\nis_whitened: {} d:{}'.format(
                videonum_as_str, lambd, auc, corr, is_whitened, d_str))
            print('Saving figure to {}'.format(plt.gcf().number))
            save_fig_to_workspace()

        except Exception as exc:
            print(exc)
            print('continuing...')

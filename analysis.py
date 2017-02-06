import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

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
    results_dirs = sorted(glob.glob('data/results/2017_02_01/*'))
    results_dirs = results_dirs[:-1]  # last one is corrupted from memory
    fignum = 0
    for results_dir in reversed(results_dirs):
        try:
            anomalousness = np.load(os.path.join(results_dir, 'anomaly_ratings.npy'))
            anomalousness = abs(anomalousness - 0.5)
            anomalousness = anomalousness/(1.0-anomalousness)
            pars = pickle.load(open(os.path.join(results_dir, 'pars.pickle'), 'r'))
            locs3 = np.load(pars.paths.files.infile_features.replace('.train', '.npy').replace('raw._','raw_locs3_'))
        except Exception as exc:
            print(exc)
            print('continuing...')

        locs3 = locs3[:-1,:]

        rs = np.arange(4, 345, 10)
        cs = np.arange(4, 634, 10)
        ts = np.arange(2, 918, 5)

        anomaly_ratings3 = np.zeros((len(rs), len(cs), len(ts)))
        for anomaly_index in range(locs3.shape[0]):
            anomaly_ratings3[locs3[anomaly_index,0] == rs, locs3[anomaly_index,1] == cs, locs3[anomaly_index,2] == ts] = anomalousness[anomaly_index]
        fignum += 1
        plt.figure(fignum); plt.clf()
        plt.plot(anomaly_ratings3.reshape(-1,1))
        videonum_as_str = os.path.basename(pars.paths.files.infile_features)[0:2]
        lambd = pars.algorithm.discriminability.lambd
        plt.title('video: {}\nlambda: {}'.format(videonum_as_str, lambd))
        print('Saving figure to {}'.format(plt.gcf().number))
        save_fig_to_workspace()

        # plt.show()

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.ndimage
import scipy.signal
import shutil
import display_pyutils

# Load the FOCUS packageimport sys
import sys
sys.path.append('/home/allie/projects/focus')  # Just to remember where this path is from!

IM_DIR = '/home/allie/workspace/images'


def apply_averaging_filter(x, filter_size=5):
    return np.convolve(signal, np.ones(filter_size,) / float(filter_size), mode='valid')


def apply_median_filter(x, filter_size=5):
    return scipy.signal.medfilt(x, filter_size)


if __name__ == '__main__':
    results_dirs = sorted(glob.glob('/home/allie/workspace/server_sync/*/*'))
    fignum = 0
    plt.close('all')
    anomalousness_to_save = []
    pars_to_save = []
    for results_dir in reversed(results_dirs):
        fignum+=1
        print(results_dir)
        print(os.path.join(results_dir, 'anomaly_ratings.npy'))
        try:
            anomalousness = np.load(os.path.join(results_dir, 'anomaly_ratings.npy'))
            signal = anomalousness/(1.0-anomalousness)
            # Smooth temporally
            signal = apply_averaging_filter(signal, 100)
            pars = pickle.load(open(os.path.join(results_dir, 'pars.pickle'), 'rb'))
            feats_file = pars.paths.files.infile_features
            plt.figure(fignum)
            plt.fill_between(range(len(signal)), signal, facecolor=display_pyutils.GOOD_COLOR_CYCLE[0], alpha=1.0) # alpha=0.5
            signal_sorted = np.sort(signal)
            bottom_ninetyfive_percent = signal_sorted[:int(np.floor(len(signal_sorted) * 0.95))]
            y_max = np.median(bottom_ninetyfive_percent) + 3*np.std(bottom_ninetyfive_percent)
            plt.ylim([0, y_max])
            videonum_as_str = os.path.basename(feats_file)
            lambd = pars.algorithm.discriminability.lambd
            max_buffer_size = pars.algorithm.discriminability.max_buffer_size
            title = 'video: {}\nlambda: {}\nmax_buffer_size:{}'.format(videonum_as_str, lambd, max_buffer_size)
            plt.title(title)
            print('Saving figure to {}.png in workspace'.format(plt.gcf().number))
            display_pyutils.save_fig_to_workspace()
            results_figure_name = os.path.join(results_dir, 'anomaly_rating.png')
            display_pyutils.savefig(results_figure_name)
            print('Saving figure to {}'.format(results_figure_name))
            thresholded_anom_results = (signal > (np.median(signal) + 2 * np.std(bottom_ninetyfive_percent))).astype(float) * signal 
            plt.clf()
            plt.fill_between(range(len(thresholded_anom_results)), thresholded_anom_results, facecolor=display_pyutils.GOOD_COLOR_CYCLE[1], alpha=1.0, label='anomalous: {:.4g}%'.format(100.0 * np.sum(thresholded_anom_results > 0) / len(thresholded_anom_results)))
            plt.legend()
            plt.ylim([0, y_max])
            plt.title(title)
            print('Saving figure to {}'.format(results_figure_name.replace('rating', 'rating_thresholded')))
            display_pyutils.savefig(results_figure_name.replace('rating', 'rating_thresholded'))
            if videonum_as_str.find('1101') != -1 and lambd == 10:
                print('results_dir of interest: {}'.format(results_dir))
                anomalousness_to_save += [anomalousness]
                pars_to_save += [pars]
                anomalous_frames = [os.path.join('/home/allie/projects/aladdin/videos/LGW_20071101_E1_CAM1frames', 'image-%06d' % frame_num + '.png') for frame_num in np.where(thresholded_anom_results > 0)[0]]
                destination_frames = [os.path.join('/home/allie/workspace/etc/1101_results', 'image-%06d' % frame_num + '.png') for frame_num in np.where(thresholded_anom_results > 0)[0]]
                for src, dest in zip(anomalous_frames, destination_frames):
                    shutil.copyfile(src, dest)
                
            video_id = '1108'
            if videonum_as_str.find(video_id) != -1 and lambd == 10:
                print('results_dir of interest: {}'.format(results_dir))
                anomalousness_to_save += [anomalousness]
                pars_to_save += [pars]
                destination_dir = '/home/allie/workspace/etc/{}_results'.format(video_id)
                os.mkdir(destination_dir)
                anomalous_frames = [os.path.join('/home/allie/projects/aladdin/videos/LGW_2007{}_E1_CAM1frames'.format(video_id), 'image-%06d' % frame_num + '.png') for frame_num in np.where(thresholded_anom_results > 0)[0]]
                destination_frames = [os.path.join(destination_dir, 'image-%06d' % frame_num + '.png') for frame_num in np.where(thresholded_anom_results > 0)[0]]
                for src, dest in zip(anomalous_frames, destination_frames):
                    shutil.copyfile(src, dest)
                
        except Exception as exc:
            print(exc)
            print('continuing...')

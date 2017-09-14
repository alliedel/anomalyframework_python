import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.signal
import shutil
import display_pyutils


def apply_averaging_filter(x, filter_size=5):
    return np.convolve(x, np.ones(filter_size,) / float(filter_size), mode='valid')


def apply_median_filter(x, filter_size=5):
    return scipy.signal.medfilt(x, filter_size)


def postprocess_signal(anomaly_ratings):
    signal = anomaly_ratings / (1 - anomaly_ratings)
    bottom_ninetyfive_percent = sorted(signal)[:int(np.floor(len(signal) * 0.95))]
    smoothed_signal = apply_averaging_filter(signal, 100)
    threshold = (np.median(signal) + 2 * np.std(bottom_ninetyfive_percent))).astype(float) * signal

    return smoothed_signal, threshold


def save_anomaly_plot(signal, pars):
    plt.figure(1); plt.clf()
    plot_anomaly_ratings(signal)
    title = 'video: {}\nlambda: {}\nmax_buffer_size:{}'.format(
        os.path.basename(pars.paths.files.infile_features), pars.algorithm.discriminability.lambd,
        pars.algorithm.discriminability.max_buffer_size)
    plt.title(title)
    print('Saving figure to {}.png in workspace'.format(plt.gcf().number))
    display_pyutils.save_fig_to_workspace()



def plot_anomaly_ratings(signal):
    plt.fill_between(range(len(signal)), signal, facecolor=display_pyutils.GOOD_COLOR_CYCLE[0],
                     alpha=1.0) # alpha=0.5
    signal_sorted = np.sort(signal)
    bottom_ninetyfive_percent = signal_sorted[:int(np.floor(len(signal_sorted) * 0.95))]
    y_max = np.median(bottom_ninetyfive_percent) + 3*np.std(bottom_ninetyfive_percent)
    plt.ylim([0, y_max])


# Given :
# - a set of anomaly ratings (continuous plus threshold or binary -- start with binary)
# - path to frames of a video
# - path to destination frames
# Output :
# - populate path to destination frames w/ video that highlights the anomaly frames (in red) /
# slows them down and speeds up non-anomalies.

def create_output_frames(anomaly_rating_binary_per_frame, input_frames, output_dir,
                         normal_fps=30*4, anomalous_fps=15):
    an_binary = anomaly_rating_binary_per_frame

    input_frames


def main():
    LOCAL_SED_VIDEO_DIR = '/home/allie/projects/aladdin/videos/'
    results_dirs = glob.glob('/home/allie/workspace/server_sync/2017_09_14/*')
    for results_dir in results_dirs:
        pars = pickle.load(open(os.path.join(results_dir, 'pars.pickle'), 'rb'))
        an = np.load(results_dir + '/anomaly_ratings.npy')
        signal, threshold = postprocess_signal(an)
        save_anomaly_plot(signal, pars)
        videoname = pars.paths.files.infile_features
        anomalous_frames = sorted(glob.glob('/home/allie/projects/aladdin/videos/{}'
                                            'frames'.format(videoname)))

        input_frames =
        create_output_frames(signal > threshold, input_frames, output_dir)



        # 'image-%06d' % frame_num + '.png')

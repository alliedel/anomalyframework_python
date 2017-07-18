import numpy as np
import scipy.ndimage


def smooth_3d(signal_3d, smooth_size=5):
    """
    Uses a uniform filter to
    """
    smooth_size = 5
    smoothed_ratings3 = scipy.ndimage.uniform_filter(signal_3d,
                                                     smooth_size)
    return smoothed_ratings3


def anomaly_1d_grid_to_3d_grid(anomalousness, locs3, rs, cs, ts):
    assert locs3.shape[0] - len(anomalousness) <= 1
    locs3 = locs3[1:, :]
    # cs = range(16)
    anomaly_ratings3 = np.zeros((len(rs), len(cs), len(ts)))
    for anomaly_index in range(locs3.shape[0]):
        anomaly_ratings3[locs3[anomaly_index, 0], locs3[anomaly_index, 1],
                         locs3[anomaly_index, 2]] = anomalousness[anomaly_index]
    return anomaly_ratings3


def anomaly_3d_to_frames(anomaly_ratings3):
    smoothed_by_frame = anomaly_ratings3.sum(axis=1).sum(axis=0)
    return smoothed_by_frame

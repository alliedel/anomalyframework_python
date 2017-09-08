import numpy as np
import unittest
import tempfile

from anomalyframework import liblinear_utils
from anomalyframework import run

ONE_BASED = 1


class TestFull(unittest.TestCase):
    def easy_2d_test(self):
        prediction, ground_truth = easy_2d_test()
        self.assertListEqual(prediction.tolist(), ground_truth.tolist())


def generate_easy_2d_example():
    X = np.ones((100,2), dtype=float)
    X[50:55,1] = 2.0 + np.random.normal(0,0.2,5)
    y = np.arange(1, X.shape[0]+1)
    ground_truth = np.zeros(y.shape)
    ground_truth[50:55] = 1
    print(len(ground_truth))
    test_features_file = tempfile.mkstemp('.train')[1]

    liblinear_utils.write(X, y, test_features_file, zero_based=False)
    return test_features_file, ground_truth


def easy_2d_test():
    infile_features, ground_truth = generate_easy_2d_example()

    a, pars = run.main(infile_features=infile_features, n_shuffles=100, window_size=1,
                       window_stride_multiplier=1.0, lambd=1e-2)
    anomaly_signal = a/(1-a)
    std_multiplier = 1.0
    prediction = anomaly_signal > np.mean(anomaly_signal) + std_multiplier * np.std(anomaly_signal)
    return prediction, ground_truth


if __name__ == '__main__':
    unittest.main()


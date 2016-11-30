import numpy as np
import matplotlib.pyplot as plt

gt = np.empty((2, 2, 3))
gt[:,:,0] = [[0, 0],
             [0, 0]]
gt[:,:,1] = [[1, 1],
             [1, 0]]
gt[:,:,2] = [[0, 0],
             [0, 0]]

detection = np.empty((2, 2, 3))
detection[:,:,0] = [[1, 0],
                    [0, 0]]
detection[:,:,1] = [[1, 1],
                    [1, 0]]
detection[:,:,2] = [[1, 0],
                    [0, 0]]

gt_pixel_1d = gt.flatten()
detection_pixel_1d = detection.flatten()

gt_frame_1d = gt.any(axis=(0,1))
detection_frame_1d = detection.sum(axis=(0,1))

frame_accuracy = np.sum(1*(gt_frame_1d == (detection_frame_1d > 0))) / len(gt_frame_1d)
pixel_accuracy = np.sum(1*(gt_pixel_1d == (detection_pixel_1d > 0))) / len(gt_pixel_1d)

print('Frame accuracy: {}'.format(frame_accuracy))  # 0.33
print('Pixel accuracy: {}'.format(pixel_accuracy))  # 0.83

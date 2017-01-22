# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:27:28 2016

@author: v-yanzwu
"""

data_set_root = 'D:\\SharedData\\Tracker_Benchmark_v1.0\\'

import numpy as np
import cv2
import matplotlib.pyplot as plt

def overlap_area(test_box, restrict_box):

    result = 0

    w = min(test_box[2], restrict_box[2]) - max(test_box[0], restrict_box[0]) + 1
    h = min(test_box[3], restrict_box[3]) - max(test_box[1], restrict_box[1]) + 1

    if w>0 and h>0:
        area = (test_box[2] - test_box[0] + 1) * (test_box[3] - test_box[1] + 1) + \
            (restrict_box[2] - restrict_box[0] + 1) * (restrict_box[3] - restrict_box[1] + 1) - w*h
        result = w * h / area;
    print result
    return result

ground_truth_boxes = np.loadtxt(data_set_root + 'Basketball\\groundtruth_rect.txt', delimiter = ',')
ground_truth_boxes[:, 2] = ground_truth_boxes[:, 0] + ground_truth_boxes[:, 2] - 1
ground_truth_boxes[:, 3] = ground_truth_boxes[:, 1] + ground_truth_boxes[:, 3] - 1
test_boxes = np.loadtxt('test_result.txt', delimiter = ' ')

result = np.zeros((2, 100))
auc = 0	
for idx in range(0, 100):
    overlap_threshold = float(idx) / 100
    overlap_areas = np.zeros(ground_truth_boxes.shape[0])

    for i in range(0, ground_truth_boxes.shape[0]):
        overlap_areas[i] = overlap_area(test_boxes[i, 1:], ground_truth_boxes[i])

    correct_boxes = test_boxes[overlap_areas[:] > overlap_threshold]
    result[0][idx] = overlap_threshold
    result[1][idx] = float(correct_boxes.shape[0]) / float(test_boxes.shape[0])
    auc += 0.01 * result[1][idx]

print auc

plt.xlabel('Overlap Threshold')
plt.ylabel('Success Rate')
plt.title('Success Plot of Basketball dataset')
plt.plot(result[0], result[1])
plt.show()
'''
rectangle_color = (0, 0, 0)

for idx in range(1, ground_truth_boxes.shape[0] + 1):
    curr_frame_path = data_set_root + 'Basketball\\img\\' + ('%04d' % idx) + '.jpg'
    curr_result_frame_path = data_set_root + 'Basketball\\result\\' + ('%04d' % idx) + '.jpg'
    img = cv2.imread(curr_frame_path)
    cv2.rectangle(img, (int(test_boxes[idx - 1][1]), int(test_boxes[idx - 1][2])), (int(test_boxes[idx - 1][3]), int(test_boxes[idx - 1][4])), rectangle_color)
    cv2.imwrite(curr_result_frame_path, img)
'''


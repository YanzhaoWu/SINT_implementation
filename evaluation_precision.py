# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 22:47:13 2017

@author: v-yanzwu
"""

data_set_root = 'D:\\SharedData\\Tracker_Benchmark_v1.0\\'

import numpy as np
import matplotlib.pyplot as plt

def distance(test_box, restrict_box):
    
    test_point = np.zeros(2)
    test_point[0] = (test_box[0] + test_box[2]) / 2
    test_point[1] = (test_box[1] + test_box[3]) / 2
    
    restrict_point = np.zeros(2)
    restrict_point[0] = (restrict_box[0] + restrict_box[2]) / 2
    restrict_point[1] = (restrict_box[1] + restrict_box[3]) / 2
    
    tmp_result = test_point - restrict_point
    #print tmp_result
    result = np.sqrt(tmp_result[0]**2 + tmp_result[1]**2)
    #print result
    return result

ground_truth_boxes = np.loadtxt(data_set_root + 'Basketball\\groundtruth_rect.txt', delimiter = ',')
ground_truth_boxes[:, 2] = ground_truth_boxes[:, 0] + ground_truth_boxes[:, 2] - 1
ground_truth_boxes[:, 3] = ground_truth_boxes[:, 1] + ground_truth_boxes[:, 3] - 1
test_boxes = np.loadtxt('test_result.txt', delimiter = ' ')

result = np.zeros((2, 50))
auc = 0	
for idx in range(0, 50):
    distance_threshold = float(idx)
    distance_value = np.zeros(ground_truth_boxes.shape[0])

    for i in range(0, ground_truth_boxes.shape[0]):
        distance_value[i] = distance(test_boxes[i, 1:], ground_truth_boxes[i])

    correct_boxes = test_boxes[distance_value[:] < distance_threshold]
    result[0][idx] = distance_threshold
    result[1][idx] = float(correct_boxes.shape[0]) / float(test_boxes.shape[0])
    auc += 0.01 * result[1][idx]

print auc

plt.xlabel('Location Error Threshold')
plt.ylabel('Precision')
plt.title('Precision Plot of Basketball dataset')
plt.plot(result[0], result[1])
plt.show()
data_set_root = 'D:\\SharedData\\'#Tracker_Benchmark_v1.0\\'

import numpy as np
import cv2
test_boxes = np.loadtxt('955box.txt', delimiter = ',')
rectangle_color = (0, 0, 0)
idx = 955
curr_frame_path = data_set_root + 'Yoda\\img\\' + ('%04d' % idx) + '.jpg'
curr_result_frame_path = ('%04d' % idx) + '.jpg'
img = cv2.imread(curr_frame_path)
for i in range(0, test_boxes.shape[1]):
    cv2.rectangle(img, (int(test_boxes[0][i]), int(test_boxes[1][i])), (int(test_boxes[2][i]), int(test_boxes[3][i])), rectangle_color)
cv2.imwrite(curr_result_frame_path, img)
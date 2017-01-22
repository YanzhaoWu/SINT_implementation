import numpy as np
import cv2
test_boxes = np.loadtxt('1.txt', delimiter = ',')
rectangle_color = (0, 0, 0)
curr_result_frame_path = '1.jpg'
img = np.ones((240, 320,3))
img = img * 255
for i in range(0, test_boxes.shape[1]):
    cv2.rectangle(img, (int(test_boxes[0][i]), int(test_boxes[1][i])), (int(test_boxes[2][i]), int(test_boxes[3][i])), rectangle_color)
cv2.imwrite(curr_result_frame_path, img)
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:18:36 2016

@author: v-yanzwu
"""

caffe_path = 'D:\\Projects\\caffe_rcnn\\'
data_set_root = 'D:\\SharedData\\Tracker_Benchmark_v1.0\\'
#first_frame_path = caffe_path + 'examples\\images\\cat_gray.jpg'

import sys
sys.path.insert(0, caffe_path + 'python')

import caffe
from sklearn.linear_model import Ridge

import numpy as np


def pre_calc_sample_regions(r, num_angles, step_size, scales=[0.7071, 1, 1.4142]):   
    num_steps= int(r / step_size)
    cos_values = np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / num_angles))
    sin_values = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / num_angles))
    
    dxdys = np.zeros((2, num_steps * num_angles + 1))
    for i in range(0, num_steps):
        offset = step_size * (i + 1)
        for j in range(0, num_angles):
            dx = offset * cos_values[j]
            dy = offset * sin_values[j]
            dxdys[0][i * num_angles + j] = dx
            dxdys[1][i * num_angles + j] = dy
    
    pre_sample_regions = np.zeros((4, (num_steps * num_angles + 1) * len(scales)))
    
    jump = num_steps * num_angles + 1
    counter = 0
    for scale in scales:
        start_idx = jump * counter
        end_idx = start_idx + jump
        counter += 1
        pre_sample_regions[0:2,  start_idx : end_idx] = dxdys
        pre_sample_regions[2, start_idx : end_idx] = scale
        pre_sample_regions[3, start_idx : end_idx] = scale
    return pre_sample_regions

def calc_sample_regions(r, num_angles, step_size, x, y, w, h, img_w, img_h, scales=[0.7071, 1, 1.4142]): # default scales
    pre_sample_regions = pre_calc_sample_regions(r, num_angles, step_size, scales)
    
    pre_sample_regions[0, : ] += x
    pre_sample_regions[1, : ] += y
    pre_sample_regions[2, : ] *= w
    pre_sample_regions[3, : ] *= h

    pre_sample_regions[2, : ] = pre_sample_regions[0, : ] + pre_sample_regions[2, : ] - 1 # x w
    pre_sample_regions[3, : ] = pre_sample_regions[1, : ] + pre_sample_regions[3, : ] - 1 # y h
    pre_sample_regions = np.round(pre_sample_regions)
    
    flags = np.logical_and(np.logical_and(np.logical_and(pre_sample_regions[0, : ] > 0, pre_sample_regions[1, : ] > 0),
                                      pre_sample_regions[2, : ] < img_w), pre_sample_regions[3, : ] < img_h)
    
    sample_regions = pre_sample_regions[:, flags]

    return sample_regions
    

def overlap_area(test_box, restrict_box):

    result = 0

    w = min(test_box[2], restrict_box[2]) - max(test_box[0], restrict_box[0]) + 1
    h = min(test_box[3], restrict_box[3]) - max(test_box[1], restrict_box[1]) + 1

    if w>0 and h>0:
        area = (test_box[2] - test_box[0] + 1) * (test_box[3] - test_box[1] + 1) + \
            (restrict_box[2] - restrict_box[0] + 1) * (restrict_box[3] - restrict_box[1] + 1) - w*h
        result = w * h / area;

    return result



# overlap threshold to select training samples for box regression
overlap_threshold = 0.6
# num of angles for angular sampling for box regression
num_angles = 20
step_size = 3
init_r = 30
image_size = 512
box_regression = 25088


caffe.set_mode_cpu()

net = caffe.Net('SINT_deploy.prototxt', 'SINT_similarity.caffemodel', caffe.TEST)

net.blobs['data'].reshape(1,3,image_size,image_size)
mean_file = np.load('ilsvrc_2012_mean.npy').mean(1).mean(1)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # eg, from (227,227,3) to (3,227,227)
transformer.set_mean('data', mean_file) # mean pixel, note there is discrepancy between 'mean image' in training and 'mean pixel' in testing
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


first_frame_path = data_set_root + 'Basketball\\img\\0001.jpg'

ground_truth_boxes = np.loadtxt(data_set_root + 'Basketball\\groundtruth_rect.txt', delimiter = ',')


first_frame = caffe.io.load_image(first_frame_path)

init_box = ground_truth_boxes[0, :].copy()

input_roi = np.zeros((1,5))
# input_roi[0][0] = 0 # indicate one roi
input_roi[0,1] = init_box[0] * image_size / first_frame.shape[1] - 1 # x
input_roi[0,2] = init_box[1] * image_size / first_frame.shape[0] - 1 # y
input_roi[0,3] = (init_box[0] + init_box[2] - 1) * image_size / first_frame.shape[1] - 1 # w
input_roi[0,4] = (init_box[1] + init_box[3] - 1) * image_size / first_frame.shape[0] - 1 # h


net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(first_frame, (image_size, image_size))) # resize image
net.blobs['rois'].reshape(1,5)
net.blobs['rois'].data[...] = input_roi
first_frame_out = net.forward()

feat_q = first_frame_out['feat_l2'].copy()

feat_q = feat_q.squeeze()


samples = calc_sample_regions(float(init_r) * first_frame.shape[1] / image_size, num_angles, 1, init_box[0], init_box[1], init_box[2], init_box[3],
                              first_frame.shape[1], first_frame.shape[0], scales=[0.7,0.8,0.9,1,1/0.9,1/0.8,1/0.7])

overlap_samples = np.zeros((1, samples.shape[1]))
init_box_restrict = init_box.copy()
# init_box_restrict 
init_box_restrict[2] = init_box_restrict[2] + init_box_restrict[0] - 1
init_box_restrict[3] = init_box_restrict[3] + init_box_restrict[1] - 1

for i in range(0, samples.shape[1]):
    overlap_samples[0][i] = overlap_area(samples[:, i], init_box_restrict)

selected_samples = samples[:,overlap_samples[0,:] > overlap_threshold]

#raise NameError('HiThere')

selected_roi = np.zeros((selected_samples.shape[1], 5))
selected_roi[:,1:] = np.transpose(selected_samples).copy()
selected_roi[:, 1] = selected_roi[:, 1] * image_size / first_frame.shape[1] - 1
selected_roi[:, 3] = selected_roi[:, 3] * image_size / first_frame.shape[1] - 1
selected_roi[:, 2] = selected_roi[:, 2] * image_size / first_frame.shape[0] - 1
selected_roi[:, 4] = selected_roi[:, 4] * image_size / first_frame.shape[0] - 1 


net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.resize_image(first_frame, (image_size, image_size))) # resize image
net.blobs['rois'].reshape(selected_samples.shape[1],5)
net.blobs['rois'].data[...] = selected_roi
first_frame_train_out = net.forward()
box_regression_feats = first_frame_train_out['feat_l2'].copy() 
box_regression_feats = box_regression_feats.squeeze()
box_regression_feats = box_regression_feats[ : , 0 : box_regression] # only conv 4 for box regression

box_regression_coordinate = selected_samples.copy()

# Middle coordinate
		
box_regression_coordinate[2, : ] = box_regression_coordinate[2, : ] - box_regression_coordinate[0, : ] + 1 # w
box_regression_coordinate[3, : ] = box_regression_coordinate[3, : ] - box_regression_coordinate[1, : ] + 1 # h

box_regression_coordinate[0, : ] = box_regression_coordinate[0, : ] + 0.5 * box_regression_coordinate[2, : ]
box_regression_coordinate[1, : ] = box_regression_coordinate[1, : ] + 0.5 * box_regression_coordinate[3, : ]
		
ground_truth_coordinate = init_box.copy()
ground_truth_coordinate[0] = ground_truth_coordinate[0] + 0.5 * ground_truth_coordinate[2]
ground_truth_coordinate[1] = ground_truth_coordinate[1] + 0.5 * ground_truth_coordinate[3]

target_x = np.divide((ground_truth_coordinate[0] - box_regression_coordinate[0, : ]), box_regression_coordinate[2, : ])
target_y = np.divide((ground_truth_coordinate[1] - box_regression_coordinate[1, : ]), box_regression_coordinate[3, : ])
target_w = np.log(np.divide(ground_truth_coordinate[2], box_regression_coordinate[2, : ]))
target_h = np.log(np.divide(ground_truth_coordinate[3], box_regression_coordinate[3, : ]))

regr_x = Ridge(alpha=1,fit_intercept=False)
regr_y = Ridge(alpha=1,fit_intercept=False)
regr_w = Ridge(alpha=1,fit_intercept=False)
regr_h = Ridge(alpha=1,fit_intercept=False)

regr_x.fit(box_regression_feats, target_x)
regr_y.fit(box_regression_feats, target_y)
regr_w.fit(box_regression_feats, target_w)
regr_h.fit(box_regression_feats, target_h)


# Handle with subsequent frames

frame_idx = range(2, ground_truth_boxes.shape[0] + 1)
previous_box = init_box

result_boxes = np.zeros((ground_truth_boxes.shape[0], 5)) # Score, x1, y1, x2, y2
result_boxes[0,0] = 3
result_boxes[0,1] = init_box[0]
result_boxes[0,2] = init_box[1]
result_boxes[0,3] = init_box[0] + init_box[2] - 1
result_boxes[0,4] = init_box[1] + init_box[3] - 1

counter = 0
for idx in frame_idx:
    
    curr_frame_path = data_set_root + 'Basketball\\img\\' + ('%04d' % idx) + '.jpg'
    print curr_frame_path
    img = caffe.io.load_image(curr_frame_path)
    img_h = img.shape[0]
    img_w = img.shape[1]
    print previous_box[0], previous_box[1], init_box[2], init_box[3]
    sample_regions = calc_sample_regions(float(init_r) * first_frame.shape[1] / image_size, 10, step_size, previous_box[0], previous_box[1], init_box[2], init_box[3],
                              img.shape[1], img.shape[0])
    num_sample_regions = sample_regions.shape[1]
    print sample_regions
    #raise NameError('HiThere')
    rois = np.zeros((num_sample_regions, 5))
    rois[:, 1:] = np.transpose(sample_regions).copy()
    rois[:, 1] = rois[:, 1] * image_size / img_w - 1
    rois[:, 3] = rois[:, 3] * image_size / img_w - 1
    rois[:, 2] = rois[:, 2] * image_size / img_h - 1
    rois[:, 4] = rois[:, 4] * image_size / img_h - 1

    net.blobs['data'].data[...] =  transformer.preprocess('data', caffe.io.resize_image(img, (image_size, image_size)))
    net.blobs['rois'].reshape(num_sample_regions, 5)
    net.blobs['rois'].data[...] = rois
    test_out = net.forward()
    test_feats = test_out['feat_l2'].copy()
    test_feats = test_feats.squeeze()
    
    scores = np.dot(test_feats, feat_q)
    
    max_idx = np.argmax(scores)
    #print 'scores'
    #print scores
    
    result_boxes[counter + 1, 0] = scores[max_idx]
    
    previous_box = sample_regions[:, max_idx].copy()
    previous_box[2] = previous_box[2] - previous_box[0] + 1 # w
    previous_box[3] = previous_box[3] - previous_box[1] + 1 # h 
    
    box_feat = test_feats[max_idx, 0 : box_regression].copy()
    pred_x = regr_x.decision_function(box_feat)
    pred_y = regr_y.decision_function(box_feat)
    pred_w = regr_w.decision_function(box_feat)
    pred_h = regr_h.decision_function(box_feat)
    
    new_x = pred_x * previous_box[2] + previous_box[0] + 0.5 * previous_box[2]
    new_y = pred_y * previous_box[3] + previous_box[1] + 0.5 * previous_box[3]
    new_w = previous_box[2] * np.exp(pred_w)
    new_h = previous_box[3] * np.exp(pred_h)
    			
    new_x = new_x - 0.5 * new_w
    new_y = new_y - 0.5 * new_h
    
    previous_box[0] = new_x
    previous_box[1] = new_y
    previous_box[2] = new_w
    previous_box[3] = new_h

    result_boxes[counter+1,1] = previous_box[0]
    result_boxes[counter+1,2] = previous_box[1]
    result_boxes[counter+1,3] = previous_box[2] + previous_box[0] - 1
    result_boxes[counter+1,4] = previous_box[3] + previous_box[1] - 1
    
    previous_box = sample_regions[:, max_idx].copy()    

    counter += 1
    
    print new_x, new_y, new_w, new_h
    
np.savetxt('test_result.txt', result_boxes, fmt = '%f')

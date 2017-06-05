'''
Transplants parameters to full conv version of net
'''

import numpy as np
import cv2
import time
import os
from operator import itemgetter

# ==================  caffe  ======================================
caffe_root = '/home/work/qinhuan/mywork/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load face_12c  ======================================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license_4c_iter_400000.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

for layer, blob in net.blobs.iteritems():
    print (layer, blob.data.shape)
for layer, param in net.params.iteritems():
    print (layer, param[0].data.shape, param[1].data.shape)
#exit(0)

params = ['fc2', 'fc3']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)


# Load the fully convolutional network to transplant the parameters.
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license4c_full_conv.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license_4c_iter_400000.caffemodel'
net_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params_full_conv = ['fc2-conv', 'fc3-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

# transplant
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    #import pdb;pdb.set_trace();
    #print fc_params[pr][0].flat
    #exit(0)
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license4c_full_conv_1x5.caffemodel')

import cv2
import os
import numpy as np
import sys

caffe_root = '/home/work/qinhuan/mywork/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
sys.path.append('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/preprocess_data/lib')
from face_detection_functions import *
from calc_recall import *

MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/license_48c_lmdb_iter_400000.caffemodel'
net_48c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

img_dir = '/home/work/qinhuan/mywork/license_plate/data/pos_48c'
#img_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_48c_qh/negative_01'
for file in os.listdir(img_dir):
    if file.endswith('.jpg'):
        img_name = img_dir + '/' + file
        img = cv2.imread(img_name)
        img = np.array(img, dtype=np.float32)
        img -= np.array((117, 103, 89))
        caffe_img_resized = cv2.resize(img, (72, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        #net_48c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48c.blobs['data'].data[0, ...] = caffe_img_resized_CHW
        net_48c.forward()
        prediction = net_48c.blobs['prob'].data
        print prediction[0][0], prediction[0][1]

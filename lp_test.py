import numpy as np
import cv2
import os
import sys
import time

caffe_root = '/home/work/qinhuan/mywork/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()
#caffe.set_mode_gpu()
#caffe.set_device(0)

sys.path.append('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/preprocess_data/lib')
from lp_detection_functions import *
from calc_recall import *

img_dir = '/home/work/data/images'
img_list_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'
img_list = open(img_list_file, 'r')
ann_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts'
# ==================  load pictures  ======================================
file_list = []      # list to save image names
for line in img_list.readlines():
    file_list.append(line.strip())

number_of_pictures = len(file_list)     # 9101 pictures
print number_of_pictures
# ==================  load lp12c_full_conv  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv_5x17.caffemodel'
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license4c_full_conv.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license4c_full_conv_1x5.caffemodel'
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load lp12cal  ==========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/license_12cal_iter_400000.caffemodel'
net_12_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load lp24c  ==========================

MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24c/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24c/license_24c_iter_400000.caffemodel'

net_24c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
#===================  load 24cal net =========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24cal/license_24cal_iter_400000.caffemodel'

net_24_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load 48 net ===========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/license_48c_lmdb_iter_400000.caffemodel'

net_48c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# ==================  load 48cal net ==========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48cal/license_48cal_iter_400000.caffemodel'

net_48_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# =========== evaluate lps ============

bao = 0
for current_picture in range(number_of_pictures):
    if (current_picture + 1) % 10 == 0:
        print 'Processing image : ' + str(current_picture)
    image_name = file_list[current_picture]
    image_file_name = img_dir + '/' + image_name

    img = cv2.imread(image_file_name)   # load image
    img_crop = img[int(0.6*img.shape[0]) : img.shape[0], :]
    min_lp_size = 15 #change
    max_lp_size = 60 #change

    if img is None:
        continue

    img_forward = np.array(img_crop, dtype=np.float32)
    # 12
    rectangles = detect_lp_12c_net(net_12c_full_conv, img_forward, min_lp_size, max_lp_size, True, 1.414, 0.90, np.array((58, 60, 58)))  
    rectangles = cal_lp_12c_net(net_12_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))      # calibration
    rectangles = local_nms(rectangles)      # apply local NMS
    # 24
    rectangles = detect_lp_24c_net(net_24c, img_forward, rectangles, 0.2, np.array((113, 96, 80)))
    #rectangles = cal_lp_24c_net(net_24_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))
    #rectangles = local_nms(rectangles)
    # 48
    #rectangles = detect_lp_48c_net(net_48c, img_forward, rectangles, 0.2, np.array((117, 103, 89)))
    rectangles = global_nms_withIoM(rectangles)
    rectangles = cal_lp_48c_net(net_48_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))
    baocun = '/home/work/qinhuan/mywork/license_plate/data/tmp'
    for rec in rectangles:
        cropped = img_forward[rec[1]:rec[3], rec[0]:rec[2]]
        cv2.imwrite(baocun + '/' + str(bao).zfill(5) + '.jpg', cropped)
        bao += 1


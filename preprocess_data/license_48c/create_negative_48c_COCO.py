'''
creates hard negatives for face_24c
'''
import numpy as np
import cv2
import os
import sys

caffe_root = '/home/work/qinhuan/mywork/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(2)

sys.path.append('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/preprocess_data/lib')
from face_detection_functions import *
from calc_recall import *
#from test_recall import IoUs

img_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_original'

# ==================  load face12c_full_conv  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv_5x17.caffemodel'
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load face12cal  ==========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/license_12cal_iter_400000.caffemodel'
net_12_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load face24c  ==========================
#MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24cCOCO_stepsize30000/deploy.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24cCOCO_stepsize30000/license_24c_iter_200000.caffemodel'

MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24c/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24c/license_24c_iter_400000.caffemodel'

net_24c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
#===================  load 24cal net =========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24cal/license_24cal_iter_400000.caffemodel'

net_24_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# ==================  load pictures  ======================================
file_list = []      # list to save image names
for file in os.listdir(img_dir):
    if file.endswith('.jpg'):
        file_list.append(file.strip())

number_of_pictures = len(file_list)     # 9101 pictures
print number_of_pictures
# =========== evaluate faces ============


save_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_48c/negative_'
save_img_number = 0

before = 0
after = 0
#number_of_pictures = 3100
for current_picture in range(0, number_of_pictures):
    if (current_picture + 1) % 10 == 0:
        print 'Processing image : ' + str(current_picture)
    image_name = file_list[current_picture]
    image_file_name = img_dir + '/' + image_name

    img = cv2.imread(image_file_name)   # load image
    min_face_size = 15 #change
    max_face_size = 60
    stride = 3

    if img is None:
        continue

    img_forward = np.array(img, dtype=np.float32)
    
    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size, max_face_size, stride, True, 1.414, 0.5, np.array((57, 59, 57)))  
    rectangles = local_nms(rectangles)      # apply local NMS
    before += len(rectangles)
    
    rectangles = detect_face_24c_net(net_24c, img_forward, rectangles, 0.00002, np.array((121, 108, 94)))
    rectangles = cal_face_24c_net(net_24_cal, img_forward, rectangles, 0.5, np.array((126, 102, 81)))
    rectangles = local_nms(rectangles)
    after += len(rectangles)

    for rec in rectangles:
        neg_sample = img[rec[1]:rec[3], rec[0]:rec[2]]
        number_dir = save_img_number / 20000 + 1
        if number_dir == 21:
            exit(0)
        save_dir_neg = save_dir + str(number_dir).zfill(2)
        if not os.path.exists(save_dir_neg):
            os.makedirs(save_dir_neg)
        save_img = save_dir_neg + '/neg' + str(number_dir).zfill(2) + '_' + str(save_img_number).zfill(6) + '.jpg'
        cv2.imwrite(save_img, neg_sample)
        save_img_number += 1 


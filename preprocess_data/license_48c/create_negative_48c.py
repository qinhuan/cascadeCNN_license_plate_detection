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

img_dir = '/home/work/data/dididata/object-v170223/images'
img_list_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'
img_list = open(img_list_file, 'r')
ann_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts_new'
#ann_dir = '/home/work/data/dididata/object-v170223/txts'

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
for img in img_list.readlines():
    file_list.append(img.strip())

number_of_pictures = len(file_list)     # 9101 pictures
print number_of_pictures
# =========== evaluate faces ============

def IoU(b1, b2):
    w1 = b1[2] - b1[0]
    h1 = b1[3] - b1[1]
    w2 = b2[2] - b2[0]
    h2 = b2[3] - b2[1]
    assert w1 >= 0 and h1 >= 0 and w2 >= 0 and h2 >= 0, 'illegal box'
    s1 = w1 * h1
    s2 = w2 * h2

    b = [0] * 4
    b[0] = max(b1[0], b2[0])
    b[1] = max(b1[1], b2[1])
    b[2] = min(b1[2], b2[2]) - b[0]
    b[3] = min(b1[3], b2[3]) - b[1]
    s = max(0, b[2]) * max(0, b[3])
    return s * 1.0 / (s1 + s2 - s)

def IoUs(crop_region, ann_list):
    for ann in ann_list:
        if IoU(crop_region, ann) > 0.1:
            return False
    return True

def Pos(crop_region, ann_list):
    for ann in ann_list:
        if IoU(crop_region, ann) > 0.5:
            return True
    return False

save_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_48c/negative_'
save_dir_pos = '/home/work/qinhuan/mywork/license_plate/data/pos_48c'
save_img_number = 0
save_pos = 0
before = 0
after = 0
#number_of_pictures = 3100
for current_picture in range(0, number_of_pictures - 3000):
    if (current_picture + 1) % 10 == 0:
        print 'Processing image : ' + str(current_picture)
    image_name = file_list[current_picture]
    image_file_name = img_dir + '/' + image_name

    img = cv2.imread(image_file_name)   # load image
    img_crop = img[int(0.6*img.shape[0]) : img.shape[0], :]
    min_face_size = 15 #change
    max_face_size = 60
    stride = 3

    if img is None:
        continue

    img_forward = np.array(img_crop, dtype=np.float32)
    
    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size, max_face_size, stride, True, 1.414, 0.5, np.array((57, 59, 57)))  
    rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))
    rectangles = local_nms(rectangles)      # apply local NMS
    before += len(rectangles)
    rectangles = detect_face_24c_net(net_24c, img_forward, rectangles, 0.00001, np.array((113, 96, 80)))
    rectangles = cal_face_24c_net(net_24_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))
    rectangles = local_nms(rectangles)
    after += len(rectangles)

    # load anno gt_ bbox
    gt_bbox = []
    prefix = image_name.split('.')[0]
    ann_file = ann_dir + '/' + prefix + '.txt'
    if os.path.exists(ann_file) is False:
        continue
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            xywh = line.strip().split(' ')
            xmin = float(xywh[0])
            ymin = float(xywh[1]) - int(0.6*img.shape[0])
            xmax = float(xywh[2]) + float(xywh[0])
            ymax = float(xywh[3]) + ymin
            # xmin = float(xywh[3])
            # ymin = float(xywh[4]) - int(0.6*img.shape[0])
            # xmax = float(xywh[5]) 
            # ymax = float(xywh[6]) - int(0.6*img.shape[0])
            if ymin <= 0:
                continue
            #if float(xywh[3]) < 20 or float(xywh[3]) > 60:
            #    continue
            gt_bbox.append([xmin, ymin, xmax, ymax])
    f.close()
   
    for rec in rectangles:
        if IoUs(rec, gt_bbox) is True:
            neg_sample = img_crop[rec[1]:rec[3], rec[0]:rec[2]]
            number_dir = save_img_number / 20000 + 1
            if number_dir == 21:
                exit(0)
            save_dir_neg = save_dir + str(number_dir).zfill(2)
            if not os.path.exists(save_dir_neg):
                os.makedirs(save_dir_neg)
            save_img = save_dir_neg + '/neg' + str(number_dir).zfill(2) + '_' + str(save_img_number).zfill(6) + '.jpg'
            cv2.imwrite(save_img, neg_sample)
            save_img_number += 1 
        elif Pos(rec, gt_bbox) is True:
            pos_sample = img_crop[rec[1]:rec[3], rec[0]:rec[2]]
            save_img = save_dir_pos + '/pos' + str(save_pos).zfill(6) + '.jpg'
            cv2.imwrite(save_img, pos_sample)
            save_pos += 1

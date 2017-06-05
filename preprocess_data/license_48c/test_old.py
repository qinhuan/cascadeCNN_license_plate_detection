'''
creates hard negatives for face_24c
'''
import numpy as np
import cv2
import os
import sys
import time

caffe_root = '/home/work/qinhuan/mywork/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

sys.path.append('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/preprocess_data/lib')
from face_detection_functions import *
from calc_recall import *
#from test_recall import IoUs

img_dir = '/home/work/data/images'
img_list_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'
img_list = open(img_list_file, 'r')
ann_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts'


# ==================  load face12c_full_conv  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv_5x17.caffemodel'
#MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_6c/license6c_full_conv.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_6c/license6c_full_conv_2x8.caffemodel'
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

#===================  load 48 net ===========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/license_48c_lmdb_iter_400000.caffemodel'

net_48c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# ==================  load 48cal net ==========================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_48cal/license_48cal_iter_400000.caffemodel'

net_48_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# ==================  load pictures  ======================================
file_list = []      # list to save image names
for line in img_list.readlines():
    file_list.append(line.strip())

number_of_pictures = len(file_list)     # 9101 pictures
print number_of_pictures
# =========== evaluate faces ============

pred_bboxes = []
gt_bboxes = []
hit_all = 0
ann_all = 0
before = 0
after = 0
number_of_pictures = 3100

bao = 0
start = time.clock()
t1 = 0
t2 = 0
t3 = 0
t4 = 0
t5 = 0
t6 = 0
t7 = 0
t8 = 0
t9 = 0

p1 = 0
p2 = 0
p3 = 0
p4 = 0
p5 = 0
p6 = 0
for current_picture in range(number_of_pictures - 3000):
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

    # caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
    # caffe_image = caffe_image[:, :, (2, 1, 0)]
    img_forward = np.array(img_crop, dtype=np.float32)
    #img_forward -= np.array((57, 59, 57))
    # 12
    s1 = time.clock()
    rectangles = detect_old_12c_net(net_12c_full_conv, img_forward, min_face_size, max_face_size, stride, True, 1.414, 0.6, np.array((57, 59, 57)))  
    t1 += time.clock() - s1
    p1 += len(rectangles)
    s1 = time.clock()
    rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))      # calibration
    t2 += time.clock() - s1
    s1 = time.clock()
    rectangles = local_nms(rectangles)      # apply local NMS
    t3 += time.clock() - s1
    p2 += len(rectangles)
    s1 = time.clock()
    # 24
    rectangles = detect_face_24c_net(net_24c, img_forward, rectangles, 0.2, np.array((113, 96, 80)))
    t4 += time.clock() - s1
    p3 += len(rectangles)
    s1 = time.clock()
    rectangles = cal_face_24c_net(net_24_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))
    t5 += time.clock() - s1
    s1 = time.clock()
    rectangles = local_nms(rectangles)
    t6 += time.clock() - s1
    p4 += len(rectangles)
    s1 = time.clock()
    # 48
    rectangles = detect_face_48c_net(net_48c, img_forward, rectangles, 0.2, np.array((117, 103, 89)))
    t7 += time.clock() - s1
    p5 += len(rectangles)
    s1 = time.clock()
    rectangles = global_nms_withIoM(rectangles)
    t8 += time.clock() - s1
    s1 = time.clock()
    rectangles = cal_face_48c_net(net_48_cal, img_forward, rectangles, 0.1, np.array((126, 102, 81)))
    t9 += time.clock() - s1
    p6 += len(rectangles)
    s1 = time.clock()
    pred_bboxes.append(rectangles)
    after += len(rectangles)
end = time.clock()
print "read: %f s" % ((end - start) / 100)
print t1,t2,t3,t4,t5,t6,t7,t8,t9
print p1,p2,p3,p4,p5,p6
'''
    # baocun = '/home/work/qinhuan/mywork/license_plate/data/tmp'
    # for rec in rectangles:
    #     cropped = img_forward[rec[1]:rec[3], rec[0]:rec[2]]
    #     cv2.imwrite(baocun + '/' + str(bao).zfill(5) + '.jpg', cropped)
    #     bao += 1

    # load anno gt_ bbox
    gt_bbox = []
    prefix = image_name.split('.')[0]
    ann_file = ann_dir + '/' + prefix + '.txt'
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            xywh = line.strip().split(' ')
            xmin = float(xywh[0])
            ymin = float(xywh[1]) - int(0.6*img.shape[0])
            xmax = float(xywh[2]) + float(xywh[0])
            ymax = float(xywh[3]) + ymin
            if ymin <= 0:
                continue
            #if float(xywh[3]) < 20 or float(xywh[3]) > 60:
            #    continue
            gt_bbox.append([xmin, ymin, xmax, ymax])
    gt_bboxes.append(gt_bbox)
    f.close()
    
    hit, ann = calc_recall([gt_bbox], [rectangles])
    hit_all += hit
    ann_all += ann
print hit_all, ann_all
print 'recall: ', float(hit_all) / float(ann_all)
print 'before:', before
print 'after:', after
'''

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
caffe.set_device(1)

sys.path.append('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/preprocess_data/lib')
from face_detection_functions import *
from calc_recall import *

img_dir = '/home/work/data/dididata/object-v170223/images'  
img_list_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'
img_list = open(img_list_file, 'r')
ann_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts'

# ==================  load face12c_full_conv  ======================================
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv_5x17.caffemodel'

#MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c_negDataFromCoco/license12c_full_conv.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c_negDataFromCoco/license12c_full_conv_5x17.caffemodel'

#MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/deploy.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license_12c_iter_400000.caffemodel'
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load face12cal  ==========================
#MODEL_FILE = '/home/work/qinhuan/mywork/caffe/models/license_12cal/deploy.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/caffe/models/license_12cal/license_12_cal_train_iter_400000.caffemodel'

MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/license_12cal_iter_400000.caffemodel'
net_12_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# ==================  load pictures  ======================================
file_list = []      # list to save image names
for line in img_list.readlines():
    file_list.append(line.strip())

number_of_pictures = len(file_list)     # 9101 pictures
print number_of_pictures
img_list.close()
# =========== evaluate faces ============
number_of_pictures = 3300

pred_bboxes = []
gt_bboxes = []
hit_all = 0
ann_all = 0
for current_picture in range(number_of_pictures - 3000):
    if (current_picture + 1) % 100 == 0:
        print 'Processing image : ' + str(current_picture + 1)
        print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    image_name = file_list[current_picture]
    image_file_name = img_dir + '/' + image_name
    #print image_file_name
    img = cv2.imread(image_file_name)   # load image
    img_crop = img[int(0.6*img.shape[0]) : img.shape[0], :]
    min_face_size = 20 #change
    max_face_size = 60 #change
    stride = 3 #change
    if img is None:
        continue
    # caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
    # caffe_image = caffe_image[:, :, (2, 1, 0)]
    img_forward = np.array(img_crop, dtype=np.float32)
    #img_forward -= np.array((110, 115, 117)) #change
    img_forward -= np.array((57, 59, 57))
    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size, 
                    max_face_size, stride, True, 1.414, 0.1)  # threshold change
    #print 'zhiqian',len(rectangles)
    rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles, 0.3)      # calibration
    rectangles = local_nms(rectangles)      # apply local NMS
    pred_bboxes.append(rectangles)
    
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
            gt_bbox.append([xmin, ymin, xmax, ymax])
    gt_bboxes.append(gt_bbox)
    f.close()
    #print 'zhihou', len(rectangles)
    #for rec in rectangles:
    #    print rec[0], rec[1],rec[2], rec[3], rec[4], rec[5]
    #print len(gt_bbox)
    #for gt in gt_bbox:
    #    print gt[0], gt[1], gt[2], gt[3]
    #exit(0)
    hit, ann = calc_recall([gt_bbox], [rectangles])
    hit_all += hit
    ann_all += ann
print hit_all, ann_all
print 'recall: ', float(hit_all) / float(ann_all)
exit(0)
recall = calc_recall(gt_bboxes, pred_bboxes) # xyxy
num = len(recall['recall'])
#print recall['recall'][-2], recall['thresh'][-2]
#exit(0)
f = open('z.txt', 'w')
for i in range(num):
    out = '{}, {}'.format(recall['recall'][i], recall['thresh'][i])
    f.write(out + '\n')
    #if recall['recall'][i] >= 0.99:
    #    print recall['recall'][i], recall['thresh'][i]
    #    break
f.close()

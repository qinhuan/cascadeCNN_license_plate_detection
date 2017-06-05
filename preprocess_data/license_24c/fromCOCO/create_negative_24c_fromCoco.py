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
#from test_recall import IoUs

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


# ==================  load pictures  ======================================
img_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_original'
file_list = []      # list to save image names
for file in os.listdir(img_dir):
    if file.endswith('.jpg'):
        file_list.append(file.strip())

number_of_pictures = len(file_list)     # 9101 pictures
print number_of_pictures
# =========== evaluate faces ============

save_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_24c_fromCoco/negative_'
ann_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts'

save_img_number = 0

for current_picture in range(0, number_of_pictures):
    if (current_picture + 1) % 10 == 0:
        print 'Processing image : ' + str(current_picture)
    image_name = file_list[current_picture]
    image_file_name = img_dir + '/' + image_name

    img = cv2.imread(image_file_name)   # load image
    min_face_size = 20 #change
    max_face_size = 60
    stride = 3

    if img is None:
        continue

    # caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
    # caffe_image = caffe_image[:, :, (2, 1, 0)]
    img_forward = np.array(img, dtype=np.float32)
    img_forward -= np.array((57, 59, 57))
    
    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size,
                    max_face_size, stride, True, 1.414, 0.1)  # detect faces
    #rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles)      # calibration
    rectangles = local_nms(rectangles)      # apply local NMS
    #print 'now', len(rectangles)
    
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

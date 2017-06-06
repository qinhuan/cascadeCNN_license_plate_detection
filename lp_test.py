import numpy as np
import cv2
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.io as io


# ========== parameter need to change ==============#
caffe_root = '/home/work/qinhuan/mywork/caffe/'  
workspace = '/home/work/qinhuan/mywork/license_plate/'
img_dir = '/home/work/data/images'
img_list_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'

min_lp_size = 15 #change
max_lp_size = 60 #change
save_res_dir = '/home/work/qinhuan/mywork/license_plate/data/tmp'

# ===========================================================
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()
#caffe.set_mode_gpu()
#caffe.set_device(0)

sys.path.append(workspace + 'cascadeCNN_license_plate_detection/preprocess_data/lib')
from lp_detection_functions import *
from calc_recall import *

# ==================  load pictures  ======================================
img_list = open(img_list_file, 'r')
file_list = []      # list to save image names
for line in img_list.readlines():
    file_list.append(line.strip())

number_of_pictures = len(file_list)     
print number_of_pictures

# ==================  load lp12c_full_conv  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv.prototxt'
#PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_12c/license12c_full_conv_5x17.caffemodel'
MODEL_FILE = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license4c_full_conv.prototxt'
PRETRAINED = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_4c/license4c_full_conv_1x5.caffemodel'
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#=================  load lp12cal  ==========================
MODEL_FILE = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/deploy.prototxt'
PRETRAINED = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_12cal/license_12cal_iter_400000.caffemodel'
net_12_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load lp24c  ==========================
MODEL_FILE = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_24c/deploy.prototxt'
PRETRAINED = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_24c/license_24c_iter_400000.caffemodel'
net_24c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load 24cal net =========================
MODEL_FILE = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_24cal/deploy.prototxt'
PRETRAINED = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_24cal/license_24cal_iter_400000.caffemodel'
net_24_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

#===================  load 48 net ===========================
MODEL_FILE = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/deploy.prototxt'
PRETRAINED = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_48c_lmdb/license_48c_lmdb_iter_400000.caffemodel'
net_48c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# ==================  load 48cal net ==========================
MODEL_FILE = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_48cal/deploy.prototxt'
PRETRAINED = workspace + 'cascadeCNN_license_plate_detection/train_net/jobs/license_48cal/license_48cal_iter_400000.caffemodel'
net_48_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# =========== detection  lps ============
save_cnt = 0
for current_picture in range(number_of_pictures):
    if (current_picture + 1) % 10 == 0:
        print 'Processing image : ' + str(current_picture)
    image_name = file_list[current_picture]
    image_file_name = img_dir + '/' + image_name

    img = cv2.imread(image_file_name)   # load image
    img_crop = img[int(0.6*img.shape[0]) : img.shape[0], :] # change if needed

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
   
    # save results
    img_show = io.imread(image_file_name)
    plt.clf()
    plt.imshow(img_show)
    plt.axis('off');
    ax = plt.gca()

    for rec in rectangles:
        x1 = rec[0]
        y1 = rec[1] + int(0.6*img.shape[0])
        x2 = rec[2]
        y2 = rec[3] + int(0.6*img.shape[0])
        colors = plt.cm.hsv(np.linspace(0, 1, 1)).tolist()
        coords = (x1, y1), x2 - x1, y2 - y1
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=colors[0], linewidth=2))
    plt.savefig(save_res_dir + '/' + str(save_cnt).zfill(5) + '.jpg', dpi=200, bbox_inches="tight")
    save_cnt += 1
    
    # for rec in rectangles:
    #     x1 = int(rec[0])
    #     y1 = int(rec[1])
    #     x2 = int(rec[2])
    #     y2 = int(rec[3])
    #     cropped = img_crop[y1:y2, x1:x2]
    #     cv2.imwrite(save_res_dir + '/' + str(save_cnt).zfill(5) + '.jpg', cropped)
    #     save_cnt += 1

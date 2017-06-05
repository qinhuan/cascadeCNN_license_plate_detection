import cv2
import os
import numpy as np

caffe_root = '/home/work/qinhuan/mywork/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)

sys.path.append('/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/preprocess_data/lib')

from face_detection_functions import *

# load net
MODEL_FILE = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24c_stepsize30000/deploy.prototxt'
PRETRAINED = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/license_24c_stepsize30000/license_24c_iter_400000.caffemodel'
net_24c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# classify
img_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives_24c/negative_'
save_img_txt = '/home/work/qinhuan/mywork/license_plate/data/data_list/filter_24cFromOrigin.txt'
write_file = open(save_img_txt, 'w')
start = 1
end = 17

for i in range(startm end + 1):
    current_dir = img_dir + str(img_dir).zfill(2)
    for file in os.listdir(current_dir):
        if file.endswith(".jpg"):
            img_name = current_dir + '/' + file
            img = cv2.imread(img_name)


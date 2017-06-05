# create positive sample, crop groundtrurh from picture

import numpy as np 
import cv2
import os

train_data_file = '/home/work/qinhuan/mywork/license_plate/data/train.txt'
test_data_file = '/home/work/qinhuan/mywork/license_plate/data/test.txt'
train_data_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate'
train_data_save_dir = '/home/work/qinhuan/mywork/license_plate/data/positives'
save_file_number = 0

save_license_plate_car = '/home/work/qinhuan/mywork/license_plate/data/license_plate_car.txt'
fout = open(save_license_plate_car, 'w')

#with open(train_data_file, 'r') as f:
for data_file in [train_data_file, test_data_file]:
    f = open(data_file, 'r')
    while True:
        line = f.readline()
        if line == '':
            break
        info = line
        line = line.strip().split('.')[0]
        img_name = train_data_dir + '/' + line + '.jpg'
        img_anno = train_data_dir + '/' + line + '.txt'
        img = cv2.imread(img_name)
        with open(img_anno, 'r') as f_anno:
            while True:
                line_anno = f_anno.readline()
                if line_anno == '':
                    break
                x, y, w, h = line_anno.strip().split(' ')
                # print w,h
                # print (img_name, float(w) / float(h))
                # continue
                x = int(float(x))
                y = int(float(y))
                w = int(float(w))
                h = int(float(h))
                if w > 0 and h > 0 and x > 0 and y > 0 and \
                        x + w < img.shape[1] and y + h < img.shape[0]:
                    fout.write(info)
                    cropped_img = img[y : y + h, x : x + w]
                    file_name = train_data_save_dir + '/pos_' + str(save_file_number).zfill(6) + ".jpg"
                    save_file_number += 1
                    cv2.imwrite(file_name, cropped_img)
        


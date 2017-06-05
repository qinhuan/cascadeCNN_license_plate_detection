import os
import cv2

neg_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives_24c_fromCoco.txt'
neg_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg24cCOCO_resize12x36.txt'
fneg = open(neg_resize, 'w')
neg_save_dir = '/home/work/qinhuan/mywork/license_plate/data/neg24cCOCO_resize_12x36'
if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)

with open(neg_file, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (36, 12))
        write_name = neg_save_dir + '/' + line[0].split('/')[-1]
        fneg.write(write_name + ' ' + line[1] + '\n')
        cv2.imwrite(write_name, img)


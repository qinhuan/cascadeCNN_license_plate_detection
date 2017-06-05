import os
import cv2

pos_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_positives.txt'
neg_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives.txt'
pos_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos_resize12x36.txt'
neg_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg_resize12x36.txt'
fpos = open(pos_resize, 'w')
fneg = open(neg_resize, 'w')
pos_save_dir = '/home/work/qinhuan/mywork/license_plate/data/pos_resize_12x36'
neg_save_dir = '/home/work/qinhuan/mywork/license_plate/data/neg_resize_12x36'

with open(pos_file, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (36, 12))
        write_name = pos_save_dir + '/' + line[0].split('/')[-1]
        fpos.write(write_name + ' ' + line[1] + '\n')
        #cv2.imwrite(write_name, img)

with open(neg_file, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (36, 12))
        write_name = neg_save_dir + '/' + line[0].split('/')[-1]
        fneg.write(write_name + ' ' + line[1] + '\n')
        cv2.imwrite(write_name, img)


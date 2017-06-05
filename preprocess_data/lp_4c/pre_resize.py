import os
import cv2

pos_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_positives.txt'
neg_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives.txt'
pos_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos_resize4x12.txt'
neg_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg_resize4x12.txt'
fpos = open(pos_resize, 'w')
fneg = open(neg_resize, 'w')
pos_save_dir = '/home/work/qinhuan/mywork/license_plate/data/pos_resize_4x12'
neg_save_dir = '/home/work/qinhuan/mywork/license_plate/data/neg_resize_2x12'

with open(pos_file, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (12, 4))
        write_name = pos_save_dir + '/' + line[0].split('/')[-1]
        fpos.write(write_name + ' ' + line[1] + '\n')
        cv2.imwrite(write_name, img)

cnt = 0
with open(neg_file, 'r') as f:
    for line in f.readlines():
        cnt += 1
        if cnt % 1000 == 0:
            print cnt
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (12, 4))
        write_name = neg_save_dir + '/' + line[0].split('/')[-1]
        fneg.write(write_name + ' ' + line[1] + '\n')
        cv2.imwrite(write_name, img)


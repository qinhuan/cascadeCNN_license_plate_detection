import os
import cv2
import random

write_train_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/train_12cal.txt'
write_train = open(write_train_name, 'w')
write_val_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/val_12cal.txt'
write_val = open(write_val_name, 'w')
calibration_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_calibrations.txt'

cal = []
with open(calibration_file, 'r') as f:
    for line in f:
        cal.append(line)
number_cal = len(cal)
print number_cal

val = []
val[0:20000] = cal[0:20000]
random.shuffle(val)
for i in range(20000):
    line = val[i].strip().split(' ')
    label = line[1]
    tmp = line[0].split('/')
    img_name = tmp[-2] + '/' + tmp[-1]
    write_val.write(img_name + ' ' + label + '\n')
write_val.close()

train = cal[20000:]
random.shuffle(train)
num_train = len(train)
print 'Total training data : ' + str(num_train)
for i in range(num_train):
    if (i + 1) % 100 == 0:
        print 'Processing image number ' + str(i + 1)
    line = train[i].strip().split(' ')
    label = line[1]
    tmp = line[0].split('/')
    img_name = tmp[-2] + '/' + tmp[-1]
    write_train.write(img_name + ' ' + label + '\n')
write_train.close()

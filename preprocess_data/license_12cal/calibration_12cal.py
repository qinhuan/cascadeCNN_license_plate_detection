import cv2
import os
import random

positives_license = '/home/work/qinhuan/mywork/license_plate/data/license_plate_car.txt'
save_dir = '/home/work/qinhuan/mywork/license_plate/data/calibration_12'
img_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate'

# create dir for 45 classes
for current_dir in range(45):
    cur_dir_name = save_dir + '/' + str(current_dir).zfill(2)
    if os.path.exists(cur_dir_name) is False:
        os.makedirs(cur_dir_name)

# read img and ann
array_img = []
array_ann = []
with open(positives_license, 'r') as f:
    while True:
        line = f.readline().strip()
        if line == '':
            break
        line = line.split('.')[0]
        array_img.append(img_dir + '/' + line + '.jpg')
        array_ann.append(img_dir + '/' + line + '.txt')
img_number = len(array_img)

# create calibration data
save_img_number = 0
for i in range(img_number):
    img = cv2.imread(array_img[i])
    if (i + 1) % 100 == 0:
        print 'Processing number ' + str(i + 1)
    with open(array_ann[i], 'r') as f:
        while True:
            line = f.readline().strip()
            if line == '':
                break
            x, y, w, h = line.split(' ')
            x = int(float(x))
            y = int(float(y))
            w = int(float(w))
            h = int(float(h))

            current_label = 0

            for cur_scale in [0.83, 0.91, 1.0, 1.10, 1.21]:
                for cur_x in [-0.17, 0, 0.17]:
                    for cur_y in [-0.17, 0, 0.17]:
                        s_n = 1 / cur_scale
                        x_n = -cur_x / cur_scale
                        y_n = -cur_y / cur_scale

                        x_temp = x - (x_n * w / s_n)
                        y_temp = y - (y_n * h / s_n)
                        w_temp = w / s_n
                        h_temp = h / s_n

                        cropped_img = img[y_temp : y_temp + h_temp, x_temp : x_temp + w_temp]
                        save_image_name = save_dir + '/' + str(current_label).zfill(2) + '/cal' + str(current_label).zfill(2) + '_' + str(save_img_number).zfill(6) + '.jpg'
                        current_label += 1
                        if (x_temp < 0) or (y_temp < 0):
                            continue
                        cv2.imwrite(save_image_name, cropped_img)
            save_img_number += 1
print 'Created ' + str(save_img_number) + ' calibration images.'
            
# create all_calibration.txt file
data_dir = '/home/work/qinhuan/mywork/license_plate/data/calibration_12/'
file_list = []

for i in range(45):
    data_dir_cal = data_dir + str(i).zfill(2)
    for file in os.listdir(data_dir_cal):
        if file.endswith('.jpg'):
            write_name = data_dir_cal + '/' + file + ' ' + str(i)
            file_list.append(write_name)

write_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_calibrations.txt'
write_file = open(write_file_name, 'w')

random.shuffle(file_list)
num_len = len(file_list)
print num_len

for i in range(num_len):
    write_file.write(file_list[i] + '\n')

write_file.close()        

# create train_12cal.txt and val_12cal.txt
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


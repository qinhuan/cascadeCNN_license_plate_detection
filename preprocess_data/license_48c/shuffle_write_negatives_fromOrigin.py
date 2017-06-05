import os
import cv2
import random

data_base_dir = "/home/work/qinhuan/mywork/license_plate/data/negatives_48c" 

start_neg_dir = 1
end_neg_dir = 5
write_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives_48c.txt'
write_file = open(write_file_name, "w")

file_list = []      # list to save image names

for current_neg_dir in range(start_neg_dir, end_neg_dir + 1):
    current_dir = data_base_dir + '/negative_' + str(current_neg_dir).zfill(2)

    for file in os.listdir(current_dir):
        if file.endswith(".jpg"):
            write_name = current_dir + '/' + file + ' ' + str(0)
            file_list.append(write_name)

random.shuffle(file_list)   # shuffle list
number_of_lines = len(file_list)
print number_of_lines

for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')
write_file.close()

# resize to 24 * 72
pos_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_positives.txt'
pos_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos48c_resize24x72.txt'
fpos = open(pos_resize, 'w')
pos_save_dir = '/home/work/qinhuan/mywork/license_plate/data/pos48c_resize_24x72'
if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)

neg_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives_48c.txt'
neg_resize = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg48c_resize24x72.txt'
fneg = open(neg_resize, 'w')
neg_save_dir = '/home/work/qinhuan/mywork/license_plate/data/neg48c_resize_24x72'
if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)

cnt = 0
with open(neg_file, 'r') as f:
    for line in f.readlines():
        cnt += 1
        if cnt % 1000 == 0:
            print 'resize process images: ', cnt
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (72, 24))
        write_name = neg_save_dir + '/' + line[0].split('/')[-1]
        fneg.write(write_name + ' ' + line[1] + '\n')
        cv2.imwrite(write_name, img)

with open(pos_file, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        img = line[0]
        img = cv2.imread(img)
        img = cv2.resize(img, (72, 24))
        write_name = pos_save_dir + '/' + line[0].split('/')[-1]
        fpos.write(write_name + ' ' + line[1] + '\n')
        cv2.imwrite(write_name, img)

# write train and test list
pos_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos48c_resize24x72.txt'
neg_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg48c_resize24x72.txt'
write_val_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/val_neg48c.txt'
write_val = open(write_val_name, 'w')

pos = []
with open(pos_file_name, 'r') as f:
    for line in f:
        pos.append(line.strip())
f.close()

neg = []
with open(neg_file_name, 'r') as f:
    for line in f:
        neg.append(line.strip())
f.close()

number_pos = len(pos)
number_neg = len(neg)

val = []
val[0:500] = pos[0:500]
val[500:1000] = neg[0:500]
random.shuffle(val)
for i in range(1000):
    source = val[i].split(' ')[0]
    img_file_name = source.split('/')[-2] + '/' + source.split('/')[-1]
    label = val[i].split(' ')[1]
    write_val.write(img_file_name + ' ' + label + '\n')
write_val.close()


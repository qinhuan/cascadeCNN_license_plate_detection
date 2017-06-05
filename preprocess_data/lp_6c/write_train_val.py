import os
import cv2
import shutil
import random


pos_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos_resize6x18.txt'
neg_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg_resize6x18.txt'
write_train_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/train_6c.txt'
write_train = open(write_train_name, "w")
write_val_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/val_6c.txt'
write_val = open(write_val_name, "w")

pos = []
with open(pos_file_name, "r") as ins:
    for line in ins:
        pos.append(line.strip())      # list of positive file names and labels

neg = []
with open(neg_file_name, "r") as ins:
    for line in ins:
        neg.append(line.strip())      # list of negative file names and labels

number_of_pos = len(pos)
number_of_neg = len(neg)

# take first 500 images of pos and neg as val
val = []
val[0:500] = pos[0:500]
val[500:1000] = neg[0:500]
random.shuffle(val)
for current_image in range(1000):
    source = val[current_image].split(' ')[0]
    image_file_name = source.split('/')[-2] + '/' + source.split('/')[-1]
    label = val[current_image].split(' ')[1] # retrieve label
    write_val.write(image_file_name + ' ' + label + '\n')      # write to val.txt
write_val.close()

# train data
train = []
train[0:number_of_pos - 500] = pos[500:]  # all positives not in val are assigned to train
train[number_of_pos - 500:] = neg[500:]     # assign 450000 negatives to train
random.shuffle(train)
number_of_train_data = len(train)

# write to train.txt
for current_image in range(number_of_train_data):
    if current_image % 1000 == 0:
        print 'Processing training data : ' + str(current_image)
    source = train[current_image].split(' ')[0]   # retrieve image file name (including directory) from train
    image_file_name = source.split('/')[-2] + '/' + source.split('/')[-1]   
    label = train[current_image].strip().split(' ')[1]   # retrieve label
    write_content = image_file_name + ' ' + label + '\n'
    write_train.write(write_content)      # write to train.txt
write_train.close()


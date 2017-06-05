import os
import random

# write train and test list
pos_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos_resize12x36.txt'
neg_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg24c_resize12x36.txt'
write_val_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/val_neg24c.txt'
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

import os
import random

data_base_dir = "/home/work/qinhuan/mywork/license_plate/data/negatives_24c" 

start_neg_dir = 1
end_neg_dir = 17
write_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives_24c.txt'
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


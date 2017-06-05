import os
import random

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

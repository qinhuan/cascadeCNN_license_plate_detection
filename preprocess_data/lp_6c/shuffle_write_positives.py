import os
import random

positives_data_dir = '/home/work/qinhuan/mywork/license_plate/data/positives'
file_list = []

write_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_positives.txt'
write_file = open(write_file_name, 'w')

for file in os.listdir(positives_data_dir):
    if file.endswith('.jpg'):
        write_name = positives_data_dir + '/' + file + ' ' + str(1)
        file_list.append(write_name)

random.shuffle(file_list)
number_of_lines = len(file_list)

for i in range(number_of_lines):
    write_file.write(file_list[i] + '\n')

write_file.close()

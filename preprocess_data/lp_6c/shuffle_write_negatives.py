import os
import random

data_base_dir = "/home/work/qinhuan/mywork/license_plate/data/negatives"     # directory containing files of positives

start_neg_dir = 1
end_neg_dir = 20
write_file_name = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_negatives.txt'

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

# write to file
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()

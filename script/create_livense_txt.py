import os
import shutil

txts_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts'
img_dir = '/home/work/data/dididata/object-v170223/images'
write_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'
fout = open(write_file, 'w')

for file in os.listdir(txts_dir):
    if file.endswith('.txt'):
        prefix = file.strip().split('.')[0]
        img_name = img_dir + '/' + prefix + '.jpg'
        if os.path.exists(img_name) is False:
            continue
        fout.write(prefix + '.jpg' + '\n')
        # shutil.copy2(img_name, save_dir)
        # f = open(txts_dir + '/' + file, 'r')
        # flag = 0
        # while True:
        #     line = f.readline()
        #     if line == '':
        #         break
        #     line = line.strip().split(' ')
        #     if line[1] == 'Car':
        #         if int(line[6]) - int(line[4]) >= 150:
        #             flag = 1
        #             break
        # if flag == 1:
        #     fout.write(img_name + '\n')
            # shutil.copy2(img_name, save_dir)

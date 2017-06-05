import os

train_file = '/home/work/qinhuan/mywork/license_plate/data/train.txt'
test_file = '/home/work/qinhuan/mywork/license_plate/data/test.txt'
save_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate_car.txt'
fout = open(save_file, 'w')
for file in [train_file, test_file]:
    with open(file, 'r') as f:
        for line in f.readlines():
            fout.write(line)
    f.close()
fout.close()

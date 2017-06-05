import os
import cv2

def IoU(b1, b2):
    w1 = b1[2] - b1[0]
    h1 = b1[3] - b1[1]
    w2 = b2[2] - b2[0]
    h2 = b2[3] - b2[1]
    assert w1 >= 0 and h1 >= 0 and w2 >= 0 and h2 >= 0, 'illegal box'
    s1 = w1 * h1
    s2 = w2 * h2

    b = [0] * 4
    b[0] = max(b1[0], b2[0])
    b[1] = max(b1[1], b2[1])
    b[2] = min(b1[2], b2[2]) - b[0]
    b[3] = min(b1[3], b2[3]) - b[1]
    s = max(0, b[2]) * max(0, b[3])
    return s * 1.0 / (s1 + s2 - s)

def IoUs(crop_region, ann_list):
    for ann in ann_list:
        if IoU(crop_region, ann) > 0.1:
            return False
    return True

# load image for crop negaive sample
img_list_file = '/home/work/qinhuan/mywork/license_plate/data/license_plate.txt'
img_dir = '/home/work/data/images'
f = open(img_list_file, 'r')
img_list = []
for line in f.readlines():
    img_list.append(img_dir + '/' + line.strip())
f.close()
img_number = len(img_list)

# save negative sample
save_dir = '/home/work/qinhuan/mywork/license_plate/data/negatives/negative_'
ann_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txt_new'
save_img_number = 0
i = 0
for i in range(0, img_number, 2):
#while i < img_number:
    if (i) % 100 == 0:
        print (i+1), 'processing images.'
    img = img_list[i]
    if os.path.exists(img) is False:
        continue
    img_name = img.split('/')[-1].split('.')[0]
    ann_name = ann_dir + '/' + img_name + '.txt'
    if os.path.exists(ann_name) is False:
        continue
    ann_list = []
    with open(ann_name, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            xmin = float(line[0])
            ymin = float(line[1])
            xmax = float(line[2]) + xmin
            ymax = float(line[3]) + ymin
            ann_list.append([xmin, ymin, xmax, ymax])
    f.close()
    img = cv2.imread(img)
    height, width, tmp =  img.shape
    now_x = 0
    now_y = int(height * 0.6) #change
    scale = 15 + (i % 11) * 5
    while now_y + scale < height:
        now_x = 0
        while now_x + scale * 3 < width:
            crop_region = [now_x, now_y, now_x + 3*scale, now_y + scale]
            #print crop_region
            if IoUs(crop_region, ann_list) is True:
                img_crop = img[now_y : now_y + scale, now_x : now_x + 3*scale]
                number_dir = save_img_number / 20000 + 1
                if number_dir == 21:
                    exit(0)
                save_dir_neg = save_dir + str(number_dir).zfill(2)
                if not os.path.exists(save_dir_neg):
                    os.makedirs(save_dir_neg)
                save_img = save_dir_neg + '/neg' + str(number_dir).zfill(2) + '_' + str(save_img_number).zfill(6) + '.jpg'
                cv2.imwrite(save_img, img_crop)
                save_img_number += 1
            now_x += 210
        now_y += 70


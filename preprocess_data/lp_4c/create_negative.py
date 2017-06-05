import numpy as np
import cv2
import os

data_base_dir = "/home/work/qinhuan/mywork/license_plate/data/negatives_original"     # file containing pictures
directory = '/home/work/qinhuan/mywork/license_plate/data/negatives/negative_'    # start of path
start_neg_dir = 1
end_neg_dir = 50
file_list = []      # list of strings storing names of pictures

for file in os.listdir(data_base_dir):
    if file.endswith(".jpg"):
        file_list.append(file)

number_of_pictures = len(file_list)     # 5546 pictures
print number_of_pictures, 'negative pictures.'

# ============== create negatives =====================================
break_flag = 0
for current_neg_dir in range(start_neg_dir, end_neg_dir + 1):
    if break_flag == 1:
        break
    save_image_number = 0
    save_dir_neg = directory + str(current_neg_dir).zfill(2)    # file to save patches
    if not os.path.exists(save_dir_neg):
        os.makedirs(save_dir_neg)

    for current_image in range((current_neg_dir - 1)*300, (current_neg_dir - 1)*300 + 300):    # take 300 images
        if current_image % 100 == 0:
            print "Processing image number " + str(current_image)
        read_img_name = data_base_dir + '/' + file_list[current_image].strip()
        img = cv2.imread(read_img_name)     # read image
        height, width, channels = img.shape

        crop_size = min(height, width) / 3  # start from half of shorter side
        # print type(crop_size), type(height)
        # exit(0)

        while crop_size >= 12:
            for start_height in range(0, height, 100):
                for start_width in range(0, width, 100):
                    if (start_width + int(crop_size * 3.5)) > width or (start_height + crop_size) > height:
                        break
                    cropped_img = img[start_height : start_height + crop_size, start_width : start_width + int(crop_size * 3.5)]
                    file_name = save_dir_neg + "/neg" + str(current_neg_dir).zfill(2) + "_" + str(save_image_number).zfill(6) + ".jpg"
                    cv2.imwrite(file_name, cropped_img)
                    save_image_number += 1
            crop_size *= 0.5

        if current_image == (number_of_pictures - 1):
            break_flag = 1
            break

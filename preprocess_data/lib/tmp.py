import numpy as np
import cv2
import time
from operator import itemgetter
# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def find_initial_scale(net_kind, min_face_size):
    '''
    :param net_kind: what kind of net (12, 24, or 48)
    :param min_face_size: minimum face size
    :return:    returns scale factor
    '''
    return float(min_face_size) / net_kind
def resize_image(img, scale):
    '''
    :param img: original img
    :param scale: scale factor
    :return:    resized image
    '''
    height, width, channels = img.shape
    new_height = int(height / scale)     # resized new height
    new_width = int(width / scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim)      # resized image
    return img_resized
def draw_rectangle(net_kind, img, face):
    '''
    :param net_kind: what kind of net (12, 24, or 48)
    :param img: image to draw on
    :param face: # list of info. in format [x, y, scale]
    :return:    nothing
    '''
    x = face[0]
    y = face[1]
    scale = face[2]
    original_x = int(x * scale)      # corresponding x and y at original image
    original_y = int(y * scale)
    original_x_br = int(x * scale + net_kind * scale)    # bottom right x and y
    original_y_br = int(y * scale + net_kind * scale)
    cv2.rectangle(img, (original_x, original_y), (original_x_br, original_y_br), (255,0,0), 2)
def IoU(rect_1, rect_2):
    '''
    :param rect_1: list in format [x11, y11, x12, y12, confidence, current_scale]
    :param rect_2:  list in format [x21, y21, x22, y22, confidence, current_scale]
    :return:    returns IoU ratio (intersection over union) of two rectangles
    '''
    x11 = rect_1[0]    # first rectangle top left x
    y11 = rect_1[1]    # first rectangle top left y
    x12 = rect_1[2]    # first rectangle bottom right x
    y12 = rect_1[3]    # first rectangle bottom right y
    x21 = rect_2[0]    # second rectangle top left x
    y21 = rect_2[1]    # second rectangle top left y
    x22 = rect_2[2]    # second rectangle bottom right x
    y22 = rect_2[3]    # second rectangle bottom right y
    x_overlap = max(0, min(x12,x22) -max(x11,x21))
    y_overlap = max(0, min(y12,y22) -max(y11,y21))
    intersection = x_overlap * y_overlap
    union = (x12-x11) * (y12-y11) + (x22-x21) * (y22-y21) - intersection
    return float(intersection) / union
def IoM(rect_1, rect_2):
    '''
    :param rect_1: list in format [x11, y11, x12, y12, confidence, current_scale]
    :param rect_2:  list in format [x21, y21, x22, y22, confidence, current_scale]
    :return:    returns IoM ratio (intersection over min-area) of two rectangles
    '''
    x11 = rect_1[0]    # first rectangle top left x
    y11 = rect_1[1]    # first rectangle top left y
    x12 = rect_1[2]    # first rectangle bottom right x
    y12 = rect_1[3]    # first rectangle bottom right y
    x21 = rect_2[0]    # second rectangle top left x
    y21 = rect_2[1]    # second rectangle top left y
    x22 = rect_2[2]    # second rectangle bottom right x
    y22 = rect_2[3]    # second rectangle bottom right y
    x_overlap = max(0, min(x12,x22) -max(x11,x21))
    y_overlap = max(0, min(y12,y22) -max(y11,y21))
    intersection = x_overlap * y_overlap
    rect1_area = (y12 - y11) * (x12 - x11)
    rect2_area = (y22 - y21) * (x22 - x21)
    min_area = min(rect1_area, rect2_area)
    return float(intersection) / min_area

def local_nms(rectangles):
    '''
    param rectangles: list of rectangles. format [xmin, ymin, xmax, ymax, confidence, current_scale]
    return: list of rectangles after local NMS
    '''
    num_rec = len(rectangles)
    result_rec = []
    batch_rec = []
    for i in range(num_rec):
        if i == 0:
            batch_rec.append(rectangles[i])
        elif rectangles[i][5] == rectangles[i - 1][5]:
            batch_rec.append(rectangles[i])
        elif rectangles[i][5] != rectangles[i - 1][5]:
            result_rec.extend(global_nms(batch_rec))
            batch_rec = []
            batch_rec.append(rectangles[i])
    if len(batch_rec) != 0:
        result_rec.extend(global_nms(batch_rec))
    return result_rec

def global_nms(rectangles):
    '''
    param rectangles: list of rectangles, format [xmin, ymin, xmax, ymax, confidence, current_scale]
    return: list of rectangles after global NMS
    '''
    rec_sorted = sorted(rectangles, key=itemgetter(4))
    rec = np.array(rec_sorted)
    if len(rec) == 0:
        return rec_sorted
    idxs = [x for x in range(len(rectangles))]
    pick = []
    x1 = np.array([x[0] for x in rec])
    y1 = np.array([x[1] for x in rec])
    x2 = np.array([x[2] for x in rec])
    y2 = np.array([x[3] for x in rec])
    area = (x2[idxs[:]] - x1[idxs[:]] + 1) * (y2[idxs[:]] - y1[idxs[:]] + 1)
    thresh = 0.3
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[idxs[:last]] + area[i] - (w * h))
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thresh)[0])))
    return rec[pick]

def global_nms_withIoM(rectangles):
    '''
    param rectangles: list of rectangles, format [xmin, ymin, xmax, ymax, confidence, current_scale]
    return: list of rectangles after global NMS
    '''
    rec_sorted = sorted(rectangles, key=itemgetter(4))
    rec = np.array(rec_sorted)
    if len(rec) == 0:
        return rec_sorted
    idxs = [x for x in range(len(rectangles))]
    pick = []
    x1 = np.array([x[0] for x in rec])
    y1 = np.array([x[1] for x in rec])
    x2 = np.array([x[2] for x in rec])
    y2 = np.array([x[3] for x in rec])
    area = (x2[idxs[:]] - x1[idxs[:]] + 1) * (y2[idxs[:]] - y1[idxs[:]] + 1)
    thresh_IoU = 0.3
    thresh_IoM = 0.3
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        IoU = (w * h) / (area[idxs[:last]] + area[i] - (w * h))
        IoM = (w * h) / np.minimum(area[idxs[:last]], area[i])
        idxs = np.delete(idxs, np.concatenate(([last], 
                    np.where((IoU > thresh_IoU) | (IoM > thresh_IoM))[0])))
    return rec[pick]

def localNMS(rectangles):
    '''
    :param rectangles:  list of rectangles, which are lists in format [x11, y11, x12, y12, confidence, current_scale],
                        sorted from highest confidence to smallest
    :return:    list of rectangles after local NMS
    '''
    rectangles = sorted(rectangles, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
    result_rectangles = rectangles[:]  # list to return
    number_of_rects = len(result_rectangles)
    threshold = 0.3     # threshold of IoU of two rectangles
    cur_rect = 0
    while cur_rect < number_of_rects - 1:     # start from first element to second last element
        rects_to_compare = number_of_rects - cur_rect - 1      # elements after current element to compare
        cur_rect_to_compare = cur_rect + 1    # start comparing with element ater current
        while rects_to_compare > 0:      # while there is at least one element after current to compare
            if (IoU(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare]) >= threshold) \
                    and (result_rectangles[cur_rect][5] == result_rectangles[cur_rect_to_compare][5]):  # scale is same

                del result_rectangles[cur_rect_to_compare]      # delete the rectangle
                number_of_rects -= 1
            else:
                cur_rect_to_compare += 1    # skip to next rectangle
            rects_to_compare -= 1
        cur_rect += 1   # finished comparing for current rectangle

    return result_rectangles
def globalNMS(rectangles):
    '''
    :param rectangles:  list of rectangles, which are lists in format [x11, y11, x12, y12, confidence, current_scale],
                        sorted from highest confidence to smallest
    :return:    list of rectangles after global NMS
    '''
    result_rectangles = rectangles[:]  # list to return
    number_of_rects = len(result_rectangles)
    threshold = 0.3     # threshold of IoU of two rectangles
    cur_rect = 0
    while cur_rect < number_of_rects - 1:     # start from first element to second last element
        rects_to_compare = number_of_rects - cur_rect - 1      # elements after current element to compare
        cur_rect_to_compare = cur_rect + 1    # start comparing with element ater current
        while rects_to_compare > 0:      # while there is at least one element after current to compare
            if IoU(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare]) >= 0.2  \
                    or ((IoM(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare]) >= threshold)
                        and (result_rectangles[cur_rect_to_compare][5] < 0.85)):  # if IoU ratio is higher than threshold
                del result_rectangles[cur_rect_to_compare]      # delete the rectangle
                number_of_rects -= 1
            else:
                cur_rect_to_compare += 1    # skip to next rectangle
            rects_to_compare -= 1
        cur_rect += 1   # finished comparing for current rectangle

    return result_rectangles

# ====== Below functions (12cal ~ 48cal) take images in style of caffe (0~1 BGR)===
def detect_face_12c(net_12c_full_conv, img, min_face_size, stride,
                    multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    :param img: image to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :param scale_factor: scale to apply for pyramid
    :param threshold: score of patch must be above this value to pass to next net
    :return:    list of rectangles after global NMS
    '''
    net_kind = 12
    rectangles = []   # list of rectangles [x11, y11, x12, y12, confidence, current_scale] (corresponding to original image)

    current_scale = find_initial_scale(net_kind, min_face_size)     # find initial scale
    caffe_img_resized = resize_image(img, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape

    while current_height > net_kind and current_width > net_kind:
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))  # switch from H x W x C to C x H x W
        # shape for input (data blob is N x C x H x W), set data
        net_12c_full_conv.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12c_full_conv.blobs['data'].data[...] = caffe_img_resized_CHW
        # run net and take argmax for prediction
        net_12c_full_conv.forward()
        out = net_12c_full_conv.blobs['prob'].data[0][1, :, :]
        # print out.shape
        out_height, out_width = out.shape

        for current_y in range(0, out_height):
            for current_x in range(0, out_width):
                # total_windows += 1
                confidence = out[current_y, current_x]  # left index is y, right index is x (starting from 0)
                if confidence >= threshold:
                    current_rectangle = [int(2*current_x*current_scale), int(2*current_y*current_scale),
                                             int(2*current_x*current_scale + net_kind*current_scale),
                                             int(2*current_y*current_scale + net_kind*current_scale),
                                             confidence, current_scale]     # find corresponding patch on image
                    rectangles.append(current_rectangle)
        if multiScale is False:
            break
        else:
            caffe_img_resized = resize_image(caffe_img_resized, scale_factor)
            current_scale *= scale_factor
            current_height, current_width, channels = caffe_img_resized.shape

    return rectangles
def cal_face_12c(net_12_cal, caffe_img, rectangles):
    '''
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = caffe_img.shape
    result = []
    all_cropped_caffe_img = []

    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image
        all_cropped_caffe_img.append(cropped_caffe_img)

    if len(all_cropped_caffe_img) == 0:
        return []

    output_all = net_12_cal.predict(all_cropped_caffe_img)   # predict through caffe

    for cur_rect in range(len(rectangles)):
        cur_rectangle = rectangles[cur_rect]
        output = output_all[cur_rect]
        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))

        result.append(cur_result)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    return result
def detect_face_24c(net_24c, caffe_img, rectangles):
    '''
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    all_cropped_caffe_img = []

    for cur_rectangle in rectangles:

        x1 = cur_rectangle[0]
        y1 = cur_rectangle[1]
        x2 = cur_rectangle[2]
        y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[y1:y2, x1:x2]     # crop image
        all_cropped_caffe_img.append(cropped_caffe_img)

    if len(all_cropped_caffe_img) == 0:
        return []

    prediction_all = net_24c.predict(all_cropped_caffe_img)   # predict through caffe

    for cur_rect in range(len(rectangles)):
        confidence = prediction_all[cur_rect][1]
        if confidence > 0.05:
            cur_rectangle = rectangles[cur_rect]
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    return result
def cal_face_24c(net_24_cal, caffe_img, rectangles):
    '''
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = caffe_img.shape
    result = []

    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image
        output = net_24_cal.predict([cropped_caffe_img])   # predict through caffe
        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))

        result.append(cur_result)

    return result
def detect_face_48c(net_48c, caffe_img, rectangles):
    '''
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    all_cropped_caffe_img = []

    for cur_rectangle in rectangles:

        x1 = cur_rectangle[0]
        y1 = cur_rectangle[1]
        x2 = cur_rectangle[2]
        y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[y1:y2, x1:x2]     # crop image
        all_cropped_caffe_img.append(cropped_caffe_img)

        prediction = net_48c.predict([cropped_caffe_img])   # predict through caffe
        confidence = prediction[0][1]

        if confidence > 0.3:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    return result
def cal_face_48c(net_48_cal, caffe_img, rectangles):
    '''
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = caffe_img.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image
        output = net_48_cal.predict([cropped_caffe_img])   # predict through caffe

        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - 1.1 * original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + 1.1 * original_h / s_change))

        result.append(cur_result)

    return result

def detect_faces(nets, img_forward, caffe_image, min_face_size, stride,
                 multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    Complete flow of face cascade detection
    :param nets: 6 nets as a tuple
    :param img_forward: image in normal style after subtracting mean pixel value
    :param caffe_image: image in style of caffe (0~1 BGR)
    :param min_face_size:
    :param stride:
    :param multiScale:
    :param scale_factor:
    :param threshold:
    :return: list of rectangles
    '''
    net_12c_full_conv = nets[0]
    net_12_cal = nets[1]
    net_24c = nets[2]
    net_24_cal = nets[3]
    net_48c = nets[4]
    net_48_cal = nets[5]

    rectangles = detect_face_12c(net_12c_full_conv, img_forward, min_face_size,
                                 stride, multiScale, scale_factor, threshold)     # detect faces
    rectangles = cal_face_12c(net_12_cal, caffe_image, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_24c(net_24c, caffe_image, rectangles)
    rectangles = cal_face_24c(net_24_cal, caffe_image, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_48c(net_48c, caffe_image, rectangles)
    rectangles = globalNMS(rectangles)      # apply global NMS
    rectangles = cal_face_48c(net_48_cal, caffe_image, rectangles)      # calibration

    return rectangles

# ========== Adjusts net to take one crop of image only during test time ==========
# ====== Below functions take images in normal style after subtracting mean pixel value===
def detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size, max_face_size, stride,
                    multiScale, scale_factor, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param img: image in caffe style to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :param scale_factor: scale to apply for pyramid
    :param threshold: score of patch must be above this value to pass to next net
    :return:    list of rectangles after global NMS
    '''
    net_kind = 4
    net_kind_w = 12
    rectangles = []   # list of rectangles [x11, y11, x12, y12, confidence, current_scale] (corresponding to original image)

    current_scale = find_initial_scale(net_kind, min_face_size)     # find initial scale
    caffe_img_resized = resize_image(img_forward, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape
    
    img_forward -= mean
    while current_height > net_kind and current_width > net_kind and net_kind*current_scale <= max_face_size:
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))  # switch from H x W x C to C x H x W
        # shape for input (data blob is N x C x H x W), set data
        net_12c_full_conv.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12c_full_conv.blobs['data'].data[...] = caffe_img_resized_CHW
        # run net and take argmax for prediction
        net_12c_full_conv.forward()
        out = net_12c_full_conv.blobs['prob'].data[0][1, :, :]
        #print out.shape
        out_height, out_width = out.shape
        # print "Shape of output after resizing " + str(caffe_img_resized.shape) + " : " + str(out.shape)
        for current_y in range(0, out_height):
            for current_x in range(0, out_width):
                # total_windows += 1
                confidence = out[current_y, current_x]  # left index is y, right index is x (starting from 0)
                #print confidence
                if confidence >= threshold:
                    current_rectangle = [int(2*current_x*current_scale), int(2*current_y*current_scale),
                                             int(2*current_x*current_scale + net_kind_w*current_scale),
                                             int(2*current_y*current_scale + net_kind*current_scale),
                                             confidence, current_scale]     # find corresponding patch on image
                    rectangles.append(current_rectangle)
        if multiScale is False:
            break
        else:
            # caffe_img_resized = resize_image(caffe_img_resized, scale_factor)
            current_scale *= scale_factor
            caffe_img_resized = resize_image(img_forward, current_scale)      # resized initial caffe image
            current_height, current_width, channels = caffe_img_resized.shape
        #print '---',12*current_scale
    img_forward += mean
    return rectangles

def detect_old_12c_net(net_12c_full_conv, img_forward, min_face_size, max_face_size, stride,
                    multiScale, scale_factor, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param img: image in caffe style to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :param scale_factor: scale to apply for pyramid
    :param threshold: score of patch must be above this value to pass to next net
    :return:    list of rectangles after global NMS
    '''
    net_kind = 12
    net_kind_w = 36
    rectangles = []   # list of rectangles [x11, y11, x12, y12, confidence, current_scale] (corresponding to original image)

    current_scale = find_initial_scale(net_kind, min_face_size)     # find initial scale
    caffe_img_resized = resize_image(img_forward, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape
    
    img_forward -= mean
    while current_height > net_kind and current_width > net_kind and 12*current_scale <= max_face_size:
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))  # switch from H x W x C to C x H x W
        # shape for input (data blob is N x C x H x W), set data
        net_12c_full_conv.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12c_full_conv.blobs['data'].data[...] = caffe_img_resized_CHW
        # run net and take argmax for prediction
        net_12c_full_conv.forward()
        out = net_12c_full_conv.blobs['prob'].data[0][1, :, :]
        #print out.shape
        out_height, out_width = out.shape
        # print "Shape of output after resizing " + str(caffe_img_resized.shape) + " : " + str(out.shape)
        for current_y in range(0, out_height):
            for current_x in range(0, out_width):
                # total_windows += 1
                confidence = out[current_y, current_x]  # left index is y, right index is x (starting from 0)
                #print confidence
                if confidence >= threshold:
                    current_rectangle = [int(2*current_x*current_scale), int(2*current_y*current_scale),
                                             int(2*current_x*current_scale + net_kind_w*current_scale),
                                             int(2*current_y*current_scale + net_kind*current_scale),
                                             confidence, current_scale]     # find corresponding patch on image
                    rectangles.append(current_rectangle)
        if multiScale is False:
            break
        else:
            # caffe_img_resized = resize_image(caffe_img_resized, scale_factor)
            current_scale *= scale_factor
            caffe_img_resized = resize_image(img_forward, current_scale)      # resized initial caffe image
            current_height, current_width, channels = caffe_img_resized.shape
        #print '---',12*current_scale
    img_forward += mean
    return rectangles

def detect_license_12c_net(net_12c, img_forward, min_face_size, max_size, stride,
                    multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    Adjusts net to take one crop of image only during test time
    :param img: image in caffe style to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :param scale_factor: scale to apply for pyramid
    :param threshold: score of patch must be above this value to pass to next net
    :return:    list of rectangles after global NMS
    '''
    net_kind = 12
    net_kind_w = 36
    rectangles = []
    current_scale = find_initial_scale(net_kind, min_face_size)     # find initial scale
    caffe_img_resized = resize_image(img_forward, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape
    while current_height > net_kind and current_width > net_kind_w:
        current_x = 0
        while current_x + net_kind_w < current_width:
            current_y = 0
            while current_y + net_kind < current_height:
                cropped_img = caffe_img_resized[current_y:current_y + net_kind, current_x:current_x + net_kind_w]
                cropped_img_CHW = cropped_img.transpose((2, 0, 1))
                net_12c.blobs['data'].reshape(1, *cropped_img_CHW.shape)
                net_12c.blobs['data'].data[...] = cropped_img_CHW
                net_12c.forward()
                
                #print net_12c.blobs['prob'].data[0][0], net_12c.blobs['prob'].data[0][1], current_scale
                
                prediction = net_12c.blobs['prob'].data[0][1]
                #print current_x, current_y
                #print  [int(current_x * current_scale), int(current_y * current_scale), int((current_x + net_kind_w) * current_scale), int((current_y + net_kind) * current_scale), prediction, current_scale]
                if prediction >= threshold:
                    current_rectangle = [int(current_x * current_scale), int(current_y * current_scale), int((current_x + net_kind_w) * current_scale), int((current_y + net_kind) * current_scale), prediction, current_scale]
                    rectangles.append(current_rectangle)
                current_y += stride
            current_x += stride
            #print current_x, current_y
        if multiScale is False:
            break
        else:
            caffe_img_resized = resize_image(caffe_img_resized, scale_factor)
            current_scale *= scale_factor
            current_height, current_width, channels = caffe_img_resized.shape
        if net_kind * current_scale > max_size:
            break
        print net_kind * current_scale

    return rectangles

def cal_face_12c_net(net_12_cal, img_forward, rectangles, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''

    height, width, channels = img_forward.shape
    result = []
    if len(rectangles) == 0:
        return result
    img_forward -= mean
    f = open('a.txt', 'a') 
    for i, cur_rectangle in enumerate(rectangles):
        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image
        caffe_img_resized = cv2.resize(cropped_caffe_img, (36, 12))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_12_cal.blobs['data'].reshape(2, 3, 12, 36)
        net_12_cal.blobs['data'].data[0, ...] = caffe_img_resized_CHW
        net_12_cal.blobs['data'].data[1, ...] = caffe_img_resized_CHW
        
        net_12_cal.forward()
        output = net_12_cal.blobs['prob'].data
        prediction = output[1]      
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold
        for tmp in indices:
            f.write(',{}'.format(tmp))
        f.write('\n')
        for tmp in cur_rectangle:
            f.write(',{}'.format(tmp))
        f.write('\n')
        number_of_cals = len(indices)   # number of calibrations larger than threshold
        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            f.write(',{}'.format(cal_label))
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17
        f.write('\n')
        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals
        f.write(',{},{},{}\n'.format(s_change, x_change, y_change))

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))

        result.append(cur_result)

    img_forward += mean
    for res in result:
        for x in res:
            f.write(',{}'.format(x))
        f.write('\n')
    f.write('\n')
    f.close()
    return result

def cal_face_12c_net_new(net_12_cal, img_forward, rectangles, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''

    height, width, channels = img_forward.shape
    result = []
    if len(rectangles) == 0:
        return result
    img_forward -= mean
    net_12_cal.blobs['data'].reshape(len(rectangles), 3, 12, 36)
    f = open('b.txt', 'a')
    for i, cur_rectangle in enumerate(rectangles):
        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image
        caffe_img_resized = cv2.resize(cropped_caffe_img, (36, 12))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_12_cal.blobs['data'].data[i, ...] = caffe_img_resized_CHW
    net_12_cal.forward()
    output = net_12_cal.blobs['prob'].data
    for i, cur_rectangle in enumerate(rectangles):
        prediction = output[i]      
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold
        for tmp in indices:
            f.write(',{}'.format(tmp))
        f.write('\n')
        for tmp in cur_rectangle:
            f.write(',{}'.format(tmp))
        f.write('\n')
        number_of_cals = len(indices)   # number of calibrations larger than threshold
        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            f.write(',{}'.format(cal_label))
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17
        f.write('\n')
        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals
        f.write(',{},{},{}\n'.format(s_change, x_change, y_change))
        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))
        
        result.append(cur_result)

    img_forward += mean
    for res in result:
        for x in res:
            f.write(',{}'.format(x))
        f.write('\n')
    f.write('\n')
    f.close()
    return result

def detect_face_24c_net(net_24c, img_forward, rectangles, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    img_forward -= mean
    for cur_rectangle in rectangles:

        x1 = int(cur_rectangle[0])
        y1 = int(cur_rectangle[1])
        x2 = int(cur_rectangle[2])
        y2 = int(cur_rectangle[3])

        cropped_caffe_img = img_forward[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (36, 12 ))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        #net_24c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        #print net_24c.blobs['data'].data[...].shape
        #import pdb;pdb.set_trace()
        net_24c.blobs['data'].data[0, ...] = caffe_img_resized_CHW
        net_24c.forward()

        prediction = net_24c.blobs['prob'].data

        confidence = prediction[0][1]
        if confidence > threshold:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)
    img_forward += mean
    return result
def filter_face_24c_net(net_24c, img_forward, rectangles):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    img_forward -= np.array((121, 108, 94))
    for cur_rectangle in rectangles:

        x1 = int(cur_rectangle[0])
        y1 = int(cur_rectangle[1])
        x2 = int(cur_rectangle[2])
        y2 = int(cur_rectangle[3])

        cropped_caffe_img = img_forward[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (12, 36))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_24c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_24c.blobs['data'].data[...] = caffe_img_resized_CHW
        net_24c.forward()

        prediction = net_24c.blobs['prob'].data

        confidence = prediction[0][1]

        if confidence < 0.5:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    return result
def cal_face_24c_net(net_24_cal, img_forward, rectangles, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = img_forward.shape
    result = []
    img_forward -= mean
    for cur_rectangle in rectangles:

        original_x1 = int(cur_rectangle[0])
        original_y1 = int(cur_rectangle[1])
        original_x2 = int(cur_rectangle[2])
        original_y2 = int(cur_rectangle[3])
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (36, 12))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        #net_24_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_24_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_24_cal.forward()

        output = net_24_cal.blobs['prob'].data

        prediction = output[0]      # (44, 1) ndarray
        
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))

        result.append(cur_result)
    img_forward += mean
    return result
def detect_face_48c_net(net_48c, img_forward, rectangles, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    img_forward -= mean
    for cur_rectangle in rectangles:

        x1 = int(cur_rectangle[0])
        y1 = int(cur_rectangle[1])
        x2 = int(cur_rectangle[2])
        y2 = int(cur_rectangle[3])

        cropped_caffe_img = img_forward[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (72, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        #net_48c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48c.blobs['data'].data[0, ...] = caffe_img_resized_CHW
        net_48c.forward()

        prediction = net_48c.blobs['prob'].data
        confidence = prediction[0][1]
        if confidence > threshold:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    img_forward += mean
    return result
def cal_face_48c_net(net_48_cal, img_forward, rectangles, threshold, mean):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    img_forward -= mean
    height, width, channels = img_forward.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = int(cur_rectangle[0])
        original_y1 = int(cur_rectangle[1])
        original_x2 = int(cur_rectangle[2])
        original_y2 = int(cur_rectangle[3])
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image
        caffe_img_resized = cv2.resize(cropped_caffe_img, (72, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        #net_48_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_48_cal.forward()

        output = net_48_cal.blobs['prob'].data

        prediction = output[0]      # (44, 1) ndarray

        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - 1.1 * original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + 1.1 * original_h / s_change))

        result.append(cur_result)
    img_forward += mean
    return result

def detect_faces_net(nets, img_forward, min_face_size, stride,
                 multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    Complete flow of face cascade detection
    :param nets: 6 nets as a tuple
    :param img_forward: image in normal style after subtracting mean pixel value
    :param min_face_size:
    :param stride:
    :param multiScale:
    :param scale_factor:
    :param threshold:
    :return: list of rectangles
    '''
    net_12c_full_conv = nets[0]
    net_12_cal = nets[1]
    net_24c = nets[2]
    net_24_cal = nets[3]
    net_48c = nets[4]
    net_48_cal = nets[5]

    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size,
                                 stride, multiScale, scale_factor, threshold)  # detect faces
    rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_24c_net(net_24c, img_forward, rectangles)
    rectangles = cal_face_24c_net(net_24_cal, img_forward, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_48c_net(net_48c, img_forward, rectangles)
    rectangles = globalNMS(rectangles)      # apply global NMS
    rectangles = cal_face_48c_net(net_48_cal, img_forward, rectangles)      # calibration

    return rectangles

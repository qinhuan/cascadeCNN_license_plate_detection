'''calculate recall
'''
OVERLAP = 0.5

def calc_iou(b1, b2):
    # calculate IoU of bounding box b1 and b2.
    w1 = b1[2] - b1[0]
    h1 = b1[3] - b1[1]
    w2 = b2[2] - b2[0]
    h2 = b2[3] - b2[1]
    assert w1 >= 0 and h1 >= 0 and w2 >= 0 and h2 >=0, 'illegal box'
    s1 = w1 * h1
    s2 = w2 * h2

    b = [0]*4
    b[0] = max(b1[0], b2[0])
    b[1] = max(b1[1], b2[1])
    b[2] = min(b1[2], b2[2]) - b[0]
    b[3] = min(b1[3], b2[3]) - b[1]
    s = max(0, b[2]) * max(0, b[3])
    return s * 1.0 / (s1 + s2 - s)

def calc_roc(pred, all_num):
    """ Calculate roc curve.

    Argv:
        pred: list of [score, is_hitted].
        all_num: number of all test sample.

    Returns:
        roc: a map of precision, recall and average precision. precision
    and recall is a list of float with equal size.
    """
    pred = sorted(pred, key=lambda x:-x[0])
    precision = [1.0]
    recall = [0.0]
    thresh = [1.0]
    hit_num = 0
    pred_num = 0
    if all_num > 0:
        for x in pred:
            pred_num += 1
            hit_num += x[1]
            precision.append(hit_num * 1.0 / pred_num)
            recall.append(hit_num * 1.0 / all_num)
            thresh.append(x[0])

    # if len(precision) == 1:
    #     precision.append(0.0)
    #     recall.append(0.0)
    # else:
    #     if precision[-2] - precision[-1] < 1e-6:
    #         x = 0
    #     else:
    #         x = precision[-1] * (recall[-1] - recall[-2]) / (precision[-2] - precision[-1])
    #     if x + recall[-1] < 1:
    #         precision.append(0.0)
    #         recall.append(x + recall[-1])

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    precision.append(0.0)
    recall.append(1.0)
    thresh.append(0.0)

    # calculate average precision.
    ap = 0
    for i in range(len(precision) - 1):
        ap += (recall[i + 1] - recall[i]) * \
                (precision[i + 1] + precision[i]) * 0.5
    return {'precision':precision, 'recall':recall, 'AP':ap, 'thresh':thresh}

def calc_recall(loc, pred_loc):
    """ Detection result of single object.

    Argv:
        loc: a list of length N(number of predict images), each element is a
    list of annotation bounding boxes for the i-th image, each box contain 4
    parameters [x, y, h, w].
        pred_loc: a list of length N(number of predict images), each element is
    a list of predict bounding boxes and score for the i-th image, box and
    score contain 5 parameters [x, y, h, w, s].

    Returns:
        roc: a map of precision, recall and average precision.
    """
    N = len(loc)
    all_num = 0
    pred_sc = []
    for i in range(N):
        # 1. Sort predict location by score.
        pred_loc[i] = sorted(pred_loc[i], key=lambda x:-x[4])
        # 2. Matching predict bounding box to annotation bounding box.
        pre_num = len(pred_loc[i])
        loc_num = len(loc[i])
        all_num += loc_num
        hitted = [False] * loc_num
        qh = 0
        for p in range(pre_num):
            max_iou = 0
            hit_idx = -1
            # Get max IoU annotation box
            for l in range(loc_num):
                iou = calc_iou(pred_loc[i][p], loc[i][l])
                if iou > qh:
                    qh = iou
                if iou > max_iou:
                    max_iou = iou
                    hit_idx = l

            score = pred_loc[i][p][-2]
            if max_iou > OVERLAP and not hitted[hit_idx]:
                hitted[hit_idx] = True
                pred_sc.append([score, 1])
            else:
                pred_sc.append([score, 0])
    # print 'qh', qh
    hit = 0
    for i in pred_sc:
        if i[1] == 1:
            hit += 1
    #exit(0)
    # 3. Get ROC curve and average precision.
    # return calc_roc(pred_sc, all_num)
    return (hit, all_num)


import torch

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)  # 每个框的面积
    area2 = box_area(boxes2)

    boxes1.to(torch.device('cuda'))
    boxes2.to(torch.device('cuda'))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def m_bbox(boxes_pred_, conf_pred, boxes_real, iou_threshold, k):
    """
    :param boxes: [N, 4] N = 25200
    :param scores: [N]
    :param iou_threshold: 0.5 取iou>iou_threshold
    :param k: 10 取前conf前k
    :return: 选出来的索引
    """

    idxs = torch.arange(0, len(boxes_pred_), 1).view(boxes_pred_[:, 0].shape).to(torch.device('cuda:0'))

    scores_obj = conf_pred
    boxes_pred = boxes_pred_[:, :4]

    scores_obj = scores_obj.reshape(-1)

    boxes_real = torch.from_numpy(boxes_real)
    boxes_real = torch.tensor(boxes_real, dtype=torch.float)
    boxes_real = boxes_real.to(torch.device('cuda'))
    keep = []  # 最终保留的结果， 在boxes中对应的索引；

    num_real = len(boxes_real)
    # idxs1 = scores_obj.argsort()  # 值从小到大的 索引

    # while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
    # 得分最大框对应的索引, 以及对应的坐标
    for single_real_box in boxes_real:
        single_real_box = single_real_box.view(1, 4)

        ious = box_iou(single_real_box, boxes_pred)
        # 取大于阈值的idxs
        # keep_ious.append(idxs[ious[0] >= iou_threshold])
        # scores_obj_tmp = scores_obj[idxs[ious[0] >= iou_threshold]]

        index = idxs[ious[0] >= iou_threshold]
        boxes_ious05 = boxes_pred_[index]
        scores_ious05 = scores_obj[index]
        # idxs_tmp = boxes_ious05.
        if len(boxes_ious05) == 0:
            continue
        # obj_conf, class_ = torch.max(boxes_ious05[:, 4:5], 1, keepdim=True)
        # class_conf, class__ = torch.max(boxes_ious05[:, 5:5+1], 1, keepdim=True)
        # class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # obj_conf = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        scores_this_real_box = scores_obj[index]

        idxs_max = scores_this_real_box.argsort()

        if len(idxs_max) > k:
            boxes_ious_conf = boxes_ious05[idxs_max[-k:]]
            scores_ious_conf = scores_ious05[idxs_max[-k:]]
        else:
            boxes_ious_conf = boxes_ious05
            scores_ious_conf = scores_ious05
        if boxes_ious_conf == None:
            num_real -= 1
        else:
            scores_ious_conf = scores_ious_conf.reshape(len(scores_ious_conf), 1)
            keep.append(torch.cat([boxes_ious_conf, scores_ious_conf], 1))

    print()
    # if num_real > 0:
    #     for n in range(0, num_real - 1):
    #         c = torch.cat([boxes_ious_conf, keep[n]],0)
    if len(keep) == 0:
        return []
    else:
        return torch.cat(keep, 0)

def m_bbox_conf(boxes_pred_, conf_pred, boxes_real, iou_threshold, conf_threshold):
    """
    :param boxes: [N, 4] N = 25200
    :param scores: [N]
    :param iou_threshold: 0.5 取iou>iou_threshold
    :param conf_threshold: 0.5 取conf>0.5的
    :return: 选出来的索引
    """

    idxs = torch.arange(0, len(boxes_pred_), 1).view(boxes_pred_[:, 0].shape).to(torch.device('cuda:0'))

    scores_obj = conf_pred
    boxes_pred = boxes_pred_[:, :4]

    scores_obj = scores_obj.reshape(-1)

    boxes_real = torch.from_numpy(boxes_real)
    boxes_real = torch.tensor(boxes_real, dtype=torch.float)
    boxes_real = boxes_real.to(torch.device('cuda'))
    keep = []  # 最终保留的结果， 在boxes中对应的索引；

    num_real = len(boxes_real)
    # idxs1 = scores_obj.argsort()  # 值从小到大的 索引

    # while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
    # 得分最大框对应的索引, 以及对应的坐标
    for single_real_box in boxes_real:
        single_real_box = single_real_box.view(1, 4)

        ious = box_iou(single_real_box, boxes_pred)

        index = idxs[ious[0] >= iou_threshold]

        scores_ious05 = scores_obj[index]

        index_conf = index[scores_ious05 >= conf_threshold]
        boxes_ious05conf05 = boxes_pred_[index_conf]
        scores_ious05conf05 = scores_obj[index_conf]
        if len(boxes_ious05conf05) == 0:
            continue

        scores_ious05conf05 = scores_ious05conf05.reshape(len(scores_ious05conf05), 1)
        keep.append(torch.cat([boxes_ious05conf05, scores_ious05conf05], 1))

    print()
    # if num_real > 0:
    #     for n in range(0, num_real - 1):
    #         c = torch.cat([boxes_ious_conf, keep[n]],0)
    if len(keep) == 0:
        return []
    else:
        return torch.cat(keep, 0)
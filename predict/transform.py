# -*- coding: utf-8 -*-
# @File : transform.py
# @Author: Runist
# @Time : 2020/4/16 15:59
# @Software: PyCharm
# @Brief: 模型预测中，需要将y_pred的特征提取，并变成原有图片的尺度

import tensorflow as tf
import config.config as cfg
from model.model import yolo_head


def parse_yolov3_output(yolo_outputs, image_shape, score_threshold, max_boxes=20):
    """
    根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
    :param yolo_outputs: yolo模型的输出
    :param image_shape: 原图片的shape
    :param score_threshold: 低于这个分数的东西框不要
    :param max_boxes: 最多的检测数目
    :return:
    """
    boxes = []
    box_scores = []

    # 对三个尺度的输出获取每个预测box坐标和box的分数，score计算为 置信度 * 类别概率
    for i in range(len(yolo_outputs)):
        mask_index = cfg.anchor_masks[i]
        _boxes, _box_scores = get_boxes_and_scores(yolo_outputs[i], cfg.anchors[mask_index], image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # 将每个特征层的结果进行堆叠
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)

    # 只留下高过标准的结果
    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)

    boxes_ = []
    scores_ = []
    classes_ = []

    # 遍历每个分类，筛选
    for c in range(cfg.num_classes):
        # 取出所有box_scores >= score_threshold的框 和 得分
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 非极大抑制，去掉box重合程度高的那一些框
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                 iou_threshold=cfg.iou_threshold)

        # 获取非极大抑制后的结果，下列三个分别是：框的位置，得分与种类
        # tf.gather(params, indices, axis=0)
        # 从params的axis维根据indices的参数值获取切片
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    return boxes_, scores_, classes_


def get_boxes_and_scores(feats, anchors, image_shape):
    """
    将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
    :param feats: yolo输出的feature map
    :param anchors: 其中一种大小的先验框（总共三种）
    :param image_shape: 原图片的shape
    :return: boxes(具体坐标) box_scores（box分数）
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, calc_loss=False)
    boxes = correct_boxes(box_xy, box_wh, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, cfg.num_classes])

    return boxes, box_scores


def correct_boxes(box_xy, box_wh, image_shape):
    """
    计算物体框预测坐标在原图中的位置和大小
    :param box_xy: 经过处理过后xy
    :param box_wh: 经过处理过后wh
    :param image_shape: 原图片的shape
    :return:
    """
    # 因为生成的grid是反过来的坐标系，所以要先调整一下
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    # 类型转换
    input_shape = tf.cast(cfg.input_shape, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)

    # 送进网络的图片是正方形的，所以不够的地方会灰色补齐，那么此时，得出来的框是基于正方形的
    # 以下操作就是将 坐标和宽高 变回成原图的下的尺度

    # tf.min(input_shape / image_shape)是拿 网络的shape / 图片原来的shape，取最小的那个 算出一个倍数
    # 然后和image_shape * 运算round是四舍五入，计算scale
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    # 通过 预测框 中心xy坐标和 宽高 然后 计算得出框的左上角 坐标和右下角 坐标
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[..., 0:1],   # y_min
        box_mins[..., 1:2],   # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]   # x_max
    ], axis=-1)

    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    # boxes的shape是 (b, 13, 13, 3, 4)

    return boxes

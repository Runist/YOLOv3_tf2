# -*- coding: utf-8 -*-
# @File : loss.py
# @Author: Runist
# @Time : 2020/4/3 13:36
# @Software: PyCharm
# @Brief:

import tensorflow as tf
import config.config as cfg
from model.model import yolo_body
import tensorflow.keras as keras
from tensorflow.keras.losses import binary_crossentropy


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def box_iou(pred_box, true_box):
    """
    用于计算每个预测框与真实框的iou
    :param pred_box: 预测框的信息 -- tensor, shape=(i1,...,iN, 4), xywh
    :param true_box: 实际框的信息 -- tensor, shape=(j, 4), xywh
    :return: iou: tensor, shape=(i1, ..., iN, j)
    """
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    pred_box = tf.expand_dims(pred_box, -2)
    pred_box_xy = pred_box[..., 0:2]
    pred_box_wh = pred_box[..., 2:4]
    pred_box_wh_half = pred_box_wh/2.
    pred_box_leftup = pred_box_xy - pred_box_wh_half
    pred_box_rightdown = pred_box_xy + pred_box_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    true_box = tf.expand_dims(true_box, 0)
    true_box_xy = true_box[..., 0:2]
    true_box_wh = true_box[..., 2:4]
    true_box_wh_half = true_box_wh/2.
    true_box_leftup = true_box_xy - true_box_wh_half
    true_box_rightdown = true_box_xy + true_box_wh_half

    # 计算重合面积
    intersect_leftup = tf.maximum(pred_box_leftup, true_box_leftup)
    intersect_rightdown = tf.minimum(pred_box_rightdown, true_box_rightdown)
    # 用右下角坐标 - 左上角坐标，如果大于0就是有重叠的，如果是0就没有重叠
    intersect_wh = tf.maximum(intersect_rightdown - intersect_leftup, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # 分别算出 预测框和实际框的面积
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # 两个总面积 - 重叠部分面积 = 并集的面积
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

    return iou


def process_pred(y_pred, anchors, calc_loss=False):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    :param y_pred: 预测数据
    :param anchors: 先验框
    :param calc_loss: 是否计算loss，该函数可以在直接预测的地方用
    :return:
        bbox: 存储了x1, y1 x2, y2的坐标 shape(b, 13, 13 ,3, 4)
        objectness: 该分类的置信度 shape(b, 13, 13 ,3, 1)
        class_probs: 存储了20个分类在sigmoid函数激活后的数值 shape(b, 13, 13 ,3, 20)
        pred_box: 把xy(中心点),wh shape(b, 13, 13 ,3, 4)
    """
    grid_size = tf.shape(y_pred)[1]
    # reshape_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    # reshape_feat = tf.reshape(y_pred, [-1, grid_size, grid_size, cfg.num_bbox, cfg.num_classes + 5])

    # tf.spilt的参数对应：2-(x,y) 2-(w,h) 1-置信度 classes=20-分类数目的得分
    box_xy, box_wh, confidence, class_probs = tf.split(y_pred, (2, 2, 1, cfg.num_classes), axis=-1)
    # 举例：box_xy (13, 13, 3, 2) 3是指三个框，2是xy，其他三个输出类似

    box_xy = tf.sigmoid(box_xy)
    confidence = tf.sigmoid(confidence)
    class_probs = tf.sigmoid(class_probs)

    # !!! grid[x][y] == (y, x)
    # sigmoid(x) + cx，为什么要生成一个grid？
    grid_y = tf.tile(tf.reshape(tf.range(grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # [gx, gy, 1, 2]
    grid = tf.cast(grid, tf.float32)

    # 先将 cfg.num_bbox（3）个 数量的先验框形状 转换成和box_wh一样，不然计算不了
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, cfg.num_bbox, 2])

    # 把xy, wh归一化成比例
    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors_tensor / cfg.input_shape[::-1]

    # 把xy,wh 合并成pred_box在最后一个维度上（axis=-1）
    pred_box = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss

    if calc_loss:
        return pred_box, grid
        # return reshape_feat, pred_box, grid

    return box_xy, box_wh, confidence, class_probs


def YoloLoss(anchors):
    def compute_loss(y_true, y_pred):
        input_shape = cfg.input_shape
        grid_shapes = tf.cast(tf.shape(y_pred)[1:3], tf.float32)

        # 1. 转换 y_pred -> bbox，预测置信度，各个分类的最后一层分数， 中心点坐标+宽高
        # y_pred: (batch_size, grid, grid, anchors * (x, y, w, h, obj, ...cls))
        pred_box, grid = process_pred(y_pred, anchors, calc_loss=True)
        pred_xy = y_pred[..., 0:2]
        pred_wh = y_pred[..., 2:4]
        pred_conf = y_pred[..., 4:5]
        pred_class = y_pred[..., 5:]

        true_xy = y_true[..., 0:2] * grid_shapes - grid
        true_wh = tf.math.log(y_true[..., 2:4] / anchors * input_shape[::-1])
        object_mask = y_true[..., 4:5]
        true_class = y_true[..., 5:]

        # 将无效区域设为0
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
        # 乘上一个比例，让小框的在total loss中有更大的占比，这个系数是个超参数，如果小物体太多，可以适当调大
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, tf.bool)

        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # object_mask_bool中，为True的值，y_true[l][b, ..., 0:4]才有效
            # 最后计算除true_box的shape[box_num, 4]
            true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
            # 计算预测框 和 真实框（归一化后的xywh在图中的比例）的交并比
            iou = box_iou(pred_box[b], true_box)
            # 计算每个true_box对应的预测的iou最大的box
            best_iou = tf.reduce_max(iou, axis=-1)
            # 如果一张图片的最大iou 都小于阈值 认为这张图片没有目标
            # 则被认为是这幅图的负样本
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < cfg.ignore_thresh, tf.float32))
            return b + 1, ignore_mask

        batch_size = tf.shape(y_pred)[0]

        # while_loop创建一个tensorflow的循环体，args:1、循环条件（b小于batch_size） 2、循环体 3、传入初始参数
        # lambda b,*args: b<m：是条件函数  b,*args是形参，b<bs是返回的结果
        _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < batch_size, loop_body, [0, ignore_mask])

        # 将每幅图的内容压缩，进行处理
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)  # 扩展维度用来后续计算loss (b,13,13,3,1,1)

        xy_loss = object_mask * box_loss_scale * tf.square(true_xy, pred_xy)
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(true_wh - pred_wh)
        object_conf = tf.nn.sigmoid_cross_entropy_with_logits(object_mask, pred_conf)
        confidence_loss = object_mask * object_conf + (1 - object_mask) * object_conf * ignore_mask
        # 预测类别损失
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_class, pred_class)

        # 各个损失求平均
        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(batch_size, tf.float32)
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(batch_size, tf.float32)
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(batch_size, tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(batch_size, tf.float32)

        return xy_loss + wh_loss + confidence_loss + class_loss
    return compute_loss

# def yolo_loss(args):
#     y_true = args[:cfg.num_bbox]
#     y_pred = args[cfg.num_bbox:]
#     input_shape = cfg.input_shape
#     grid_shapes = [tf.cast(tf.shape(y_pred[i])[1:3], tf.float32) for i in range(cfg.num_bbox)]
#     loss = 0
#
#     for i in range(cfg.num_bbox):
#         # 1. 转换 y_pred -> bbox，预测置信度，各个分类的最后一层分数， 中心点坐标+宽高
#         # y_pred: (batch_size, grid, grid, anchors * (x, y, w, h, obj, ...cls))
#         mask_index = cfg.anchor_masks[i]
#         pred_box, grid = process_pred(y_pred[i], cfg.anchors[mask_index], calc_loss=True)
#         pred_xy = y_pred[i][..., 0:2]
#         pred_wh = y_pred[i][..., 2:4]
#         pred_conf = y_pred[i][..., 4:5]
#         pred_class = y_pred[i][..., 5:]
#
#         # 获取true_xy, true_wh, 获取置信度 和 分类
#         true_xy = y_true[i][..., 0:2] * grid_shapes[i] - grid
#         true_wh = tf.math.log(y_true[i][..., 2:4] / cfg.anchors[mask_index] * input_shape[::-1])
#         object_mask = y_true[i][..., 4:5]
#         true_class = y_true[i][..., 5:]
#
#         # 将无效区域设为0
#         true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
#         # 乘上一个比例，让小框的在total loss中有更大的占比，这个系数是个超参数，如果小物体太多，可以适当调大
#         box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]
#
#         # 找到负样本群组，第一步是创建一个数组，[]
#         ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
#         object_mask_bool = tf.cast(object_mask, tf.bool)
#
#         # 对每一张图片计算ignore_mask
#         def loop_body(b, ignore_mask):
#             # object_mask_bool中，为True的值，y_true[l][b, ..., 0:4]才有效
#             # 最后计算除true_box的shape[box_num, 4]
#             true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
#             # 计算预测框 和 真实框（归一化后的xywh在图中的比例）的交并比
#             iou = box_iou(pred_box[b], true_box)
#             # 计算每个true_box对应的预测的iou最大的box
#             best_iou = tf.reduce_max(iou, axis=-1)
#             # 如果一张图片的最大iou 都小于阈值 认为这张图片没有目标
#             # 则被认为是这幅图的负样本
#             ignore_mask = ignore_mask.write(b, tf.cast(best_iou < cfg.ignore_thresh, tf.float32))
#             return b + 1, ignore_mask
#
#         batch_size = tf.shape(y_pred[0])[0]
#
#         # while_loop创建一个tensorflow的循环体，args:1、循环条件（b小于batch_size） 2、循环体 3、传入初始参数
#         # lambda b,*args: b<m：是条件函数  b,*args是形参，b<bs是返回的结果
#         _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < batch_size, loop_body, [0, ignore_mask])
#
#         # 将每幅图的内容压缩，进行处理
#         ignore_mask = ignore_mask.stack()
#         ignore_mask = tf.expand_dims(ignore_mask, -1)  # 扩展维度用来后续计算loss (b,13,13,3,1,1)
#
#         xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_xy, logits=pred_xy)
#         wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(true_wh - pred_wh)
#         object_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
#         confidence_loss = object_mask * object_conf + (1 - object_mask) * object_conf * ignore_mask
#         # 预测类别损失
#         class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class, logits=pred_class)
#
#         # 各个损失求平均
#         xy_loss = tf.reduce_sum(xy_loss) / tf.cast(batch_size, tf.float32)
#         wh_loss = tf.reduce_sum(wh_loss) / tf.cast(batch_size, tf.float32)
#         confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(batch_size, tf.float32)
#         class_loss = tf.reduce_sum(class_loss) / tf.cast(batch_size, tf.float32)
#
#         loss += xy_loss + wh_loss + confidence_loss + class_loss
#     return loss


if __name__ == '__main__':
    import numpy as np

    inputs = tf.keras.layers.Input(shape=(416, 416, 3), dtype="float32")

    model = yolo_body()



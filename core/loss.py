# -*- coding: utf-8 -*-
# @File : loss.py
# @Author: Runist
# @Time : 2020/4/3 13:36
# @Software: PyCharm
# @Brief: Yolov3中计算loss

import tensorflow as tf
import config.config as cfg
from model.model import yolo_body, yolo_head


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
    pred_box_wh_half = pred_box_wh / 2.
    pred_box_leftup = pred_box_xy - pred_box_wh_half
    pred_box_rightdown = pred_box_xy + pred_box_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    true_box = tf.expand_dims(true_box, 0)
    true_box_xy = true_box[..., 0:2]
    true_box_wh = true_box[..., 2:4]
    true_box_wh_half = true_box_wh / 2.
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


def YoloLoss(anchors, summary_writer=None, optimizer=None):
    def compute_loss(y_true, y_pred):
        input_shape = cfg.input_shape
        grid_shapes = tf.cast(tf.shape(y_pred)[1:3], tf.float32)

        # 1. 转换 y_pred -> bbox，预测置信度，各个分类的最后一层分数， 中心点坐标+宽高
        # y_pred: (batch_size, grid, grid, anchors * (x, y, w, h, obj, ...cls))
        pred_xywh, grid = yolo_head(y_pred, anchors, calc_loss=True)
        pred_xy = y_pred[..., 0:2]
        pred_wh = y_pred[..., 2:4]
        pred_conf = y_pred[..., 4:5]
        pred_class = y_pred[..., 5:]

        true_xy = y_true[..., 0:2] * grid_shapes - grid
        true_wh = tf.math.log(y_true[..., 2:4] / anchors * input_shape)
        object_mask = y_true[..., 4:5]
        true_class = y_true[..., 5:]

        # 将无效区域设为0，因为y_true中没有目标的区域设置为0，Log(0)将会导致原本0区域无穷小，也就是-Inf
        # where就是即判断，condition is True 就全0，False就为原值，所以就将原值保留，-inf变为0
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
            iou = box_iou(pred_xywh[b], true_box)

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

        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(true_xy, pred_xy)
        wh_loss = object_mask * box_loss_scale * tf.square(true_wh - pred_wh)
        object_conf = tf.nn.sigmoid_cross_entropy_with_logits(object_mask, pred_conf)
        confidence_loss = object_mask * object_conf + (1 - object_mask) * object_conf * ignore_mask
        # 预测类别损失
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_class, pred_class)

        # 各个损失求平均
        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(batch_size, tf.float32)
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(batch_size, tf.float32)
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(batch_size, tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(batch_size, tf.float32)

        if summary_writer:
            # 保存到tensorboard里
            with summary_writer.as_default():
                tf.summary.scalar('xy_loss', xy_loss, step=optimizer.iterations)
                tf.summary.scalar('wh_loss', wh_loss, step=optimizer.iterations)
                tf.summary.scalar('confidence_loss', confidence_loss, step=optimizer.iterations)
                tf.summary.scalar('class_loss', class_loss, step=optimizer.iterations)

        return xy_loss + wh_loss + confidence_loss + class_loss
    return compute_loss


if __name__ == '__main__':
    import numpy as np

    inputs = tf.keras.layers.Input(shape=(416, 416, 3), dtype="float32")

    model = yolo_body()



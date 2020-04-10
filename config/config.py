# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/4/3 10:59
# @Software: PyCharm
# @Brief:
import numpy as np
import os


def get_classes(path):
    """
    加载 分类信息
    :param path: 文本路径
    :return: 分类数据
    """
    with open(path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(path):
    """
    加载先验框数据
    :param path: 文本路径
    :return: 先验框数据
    """
    with open(path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# 标签的位置
annotation_path = "./config/2012_train.txt"
# 获取classes和anchor的位置
classes_path = './config/voc_classes.txt'
anchors_path = './config/voc_anchor.txt'
# 预训练模型的位置
weights_path = 'yolo_weights.h5'
# 训练集和测试集的比例
valid_rate = 0.1
batch_size = 4
# 相关信息
input_shape = (416, 416)
num_bbox = 3
# 训练信息
epochs = 50
learn_rating = 1e-3

# 获得classes和anchor
class_names = get_classes(classes_path)
# anchors = get_anchors(anchors_path)

# 一共有多少类
num_classes = len(class_names)

output_shape = num_bbox * (num_classes + 4 + 1)

# 置信度忽略阈值
ignore_thresh = 0.5

# 先验框信息
anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
                   np.float32)

anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


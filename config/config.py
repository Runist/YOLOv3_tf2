# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/4/3 10:59
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np

# 标签的位置
annotation_path = "./config/2012_train.txt"
# 获取classes和anchor的位置
classes_path = 'voc_classes.txt'
anchors_path = './config/voc_anchor.txt'
# 预训练模型的位置
weights_path = 'yolo_weights.h5'

# 训练集和测试集的比例
valid_rate = 0.1
batch_size = 2
shuffle_size = 2

# 相关信息
input_shape = (416, 416)
num_bbox = 3

# 训练信息
epochs = 50
learn_rating = 1e-3

# 获得分类名
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# 类别总数
num_classes = len(class_names)

# iou忽略阈值
ignore_thresh = 0.5
iou_threshold = 0.3
# 分数的阈值（只留下高过这个阈值的box）
score_threshold = 0.3

# 先验框信息
# anchors = np.array([(10, 13), (16, 30), (33, 23),
#                     (30, 61), (62, 45), (59, 119),
#                     (116, 90), (156, 198), (373, 326)],
#                    np.float32)

anchors = np.array([(39, 46), (86, 119), (111, 149),
                    (171, 168), (215, 218), (284, 287),
                    (298, 311), (411, 327), (450, 438)],
                   np.float32)

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# logs path
log_dir = "./logs/summary/"
# model path
model_path = r"D:\Python_Code\YOLOv3_tf2\logs\model\yolo_test.ckpt"

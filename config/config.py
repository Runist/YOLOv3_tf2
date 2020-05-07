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
anchors_path = './config/voc_anchors.txt'
# 预训练模型的位置
pretrain_weights_path = r"D:\Python_Code\YOLOv3_tf2\config\convert_yolov3.h5"
# 是否预训练
pretrain = True
# 训练的方式
train_mode = "fit"

# 训练集和测试集的比例
valid_rate = 0.1
batch_size = 4
shuffle_size = 2

# 网络输入层信息
input_shape = (416, 416)
# 预测框的数量
num_bbox = 3

# 训练信息
epochs = 50
# 学习率
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
score_threshold = 0.1

# 先验框信息
anchors = np.array([(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)],
                   np.float32)

# anchors = np.array([(24, 32), (37, 76), (76, 54),
#                     (65, 135), (143, 114), (112, 216),
#                     (301, 171), (194, 288), (373, 332)],
#                    np.float32)

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# logs path
log_dir = r".\logs\summary"
# model path
model_path = r"D:\Python_Code\YOLOv3_tf2\logs\model\yolo_test.h5"

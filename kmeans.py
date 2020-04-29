# -*- coding: utf-8 -*-
# @File : kmeans.py
# @Author: Runist
# @Time : 2020/4/22 15:24
# @Software: PyCharm
# @Brief: K-Means计算先验框，和之前理解的不太一样，yolo的kmeans条件是需要考虑iou的


import numpy as np
import cv2 as cv
import os
import config.config as cfg


def txt2boxes(file_path):
    """
    从train.txt中取出box的相关信息，做成numpy
    :param file_path: 文件路径
    :return: np.array
    """
    f = open(file_path, 'r')
    dataSet = []
    for line in f:
        infos = line.split(" ")
        length = len(infos)
        for i in range(1, length):
            width = int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])
            height = int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])
            dataSet.append([width, height])
    result = np.array(dataSet)
    f.close()

    return result


def iou(boxes, clusters, k):  # 1 box -> k clusters
    n = boxes.shape[0]

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    # 将类中心9个数据平铺，在x轴上复制n份
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    # 选取box和cluster较小的边
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

    # 计算交并比
    inter_area = np.multiply(min_w_matrix, min_h_matrix)
    result = inter_area / (box_area + cluster_area - inter_area)

    return result


def avg_iou(boxes, clusters, k):
    accuracy = np.mean([np.max(iou(boxes, clusters, k), axis=1)])
    return accuracy


def kmeans(boxes, k, dist=np.median):
    """
    yolo的kmeans方法
    :param boxes: 所有box的坐标
    :param k: 分的类
    :param dist:
    :return:
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))

    clusters = boxes[np.random.choice(box_number, k, replace=False)]  # 初始化选取类中心
    while True:

        distances = 1 - iou(boxes, clusters, k)
        current_nearest = np.argmin(distances, axis=1)

        # 用 “==” 判断两个array 是否相同，返回的是True或False，再用.all方法判断是否全等。
        if (last_nearest == current_nearest).all():
            # 聚类中心不再更新，退出
            break

        for cluster in range(k):
            # 更新类中心
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def result2txt(data):
    """
    转换成txt文档
    :param data:
    :return:
    """
    f = open(cfg.anchors_path, 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()


if __name__ == '__main__':
    cluster_number = 9
    all_boxes = txt2boxes(cfg.annotation_path)
    result = kmeans(all_boxes, cluster_number)
    # 排序，以行为准排序，不会改变顺序

    anchors = sorted(result.tolist(), key=(lambda x: x[0] + x[1]))
    result2txt(anchors)
    print("K anchors:\n {}".format(anchors))
    print("Accuracy: {:.2f}%".format(avg_iou(all_boxes, result, cluster_number) * 100))





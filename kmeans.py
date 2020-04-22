# -*- coding: utf-8 -*-
# @File : kmeans.py
# @Author: Runist
# @Time : 2020/4/22 15:24
# @Software: PyCharm
# @Brief: K-Means计算先验框


import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from xml.etree import ElementTree
from sklearn.cluster import KMeans
import config.config as cfg


def get_wh(annotations_path):
    """
    获取box的宽高信息
    :param annotations_path: xml路径
    :return: box的宽高信息，np的结构
    """
    all_wh = []
    xml_path = os.listdir(annotations_path)

    for path in xml_path:
        path = os.path.join(annotations_path, path)

        single_box = extract_wh(path)
        # 把混在一起的wh拆成一个一个
        for single_wh in single_box:
            all_wh.append(single_wh)

    return np.array(all_wh, dtype=np.float32)


def extract_wh(xml_path):
    """
    提取一个xml中的box宽高信息
    :param xml_path: xml的路径
    :return: boxes：存有所有wh的信息
    """
    # 加载要解析的文件
    tree = ElementTree.parse(xml_path)
    # 获取文档的首部，可以理解为 数据结构中树的结构
    root = tree.getroot()
    # 提取每个边界框的信息
    boxes = list()

    # 然后用类似BeautifulSoup的findall()以Xpath语法查找，这是会返回一个列表，可以方便遍历
    for obj in root.findall('object'):
        name = obj.find('name').text

        # 不在识别类别里的不要，识别难度=1的也不要
        if name not in cfg.class_names:
            continue

        try:
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
        except:
            continue

        w, h = (xmax - xmin), (ymax - ymin)
        coors = [w, h]

        boxes.append(coors)

    return boxes


def kmeans(data, num=9):
    # 定义结束条件和中心选择的方法
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 重复试验10次，取最好的结果
    ret, label, center = cv.kmeans(data, num, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    result = [None] * num

    for i in range(num):
        # 根据索引提取出不同类
        result[i] = data[label.ravel() == i]
        # 给点画上不同颜色
        plt.scatter(result[i][:, 0], result[i][:, 1])

    # 绘制中心坐标
    plt.scatter(center[:, 0], center[:, 1], s=80, c='black', marker='d')
    plt.xlabel('Width'), plt.ylabel('Height')
    plt.show()

    """
    km = KMeans(n_clusters=9, init="k-means++", n_init=10, max_iter=3000000, tol=1e-3, random_state=0)
    label = km.fit_predict(wh)
    result = [None] * 9

    for i in range(9):
        # 根据索引提取出不同类
        result[i] = wh[label.ravel() == i]
        # 给点画上不同颜色
        plt.scatter(result[i][:, 0], result[i][:, 1])

    anchors = np.sort(km.cluster_centers_, axis=0, kind="quicksort")
    print(anchors)
    """

    return center


if __name__ == '__main__':
    wh = get_wh(r"D:\Python_Code\Tensorflow2.0\YOLOv3\VOCdevkit\VOC2012\Annotations")

    anchors = kmeans(wh)
    anchors = np.sort(anchors, axis=0, kind="quicksort")
    np.savetxt("voc_anchors.txt", anchors, fmt='%d', delimiter=',')



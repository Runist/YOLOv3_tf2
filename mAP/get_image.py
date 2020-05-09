# -*- coding: utf-8 -*-
# @File : get_image.py
# @Author: Runist
# @Time : 2020/5/9 13:16
# @Software: PyCharm
# @Brief: 复制图片到images-optional文件下，实现mAP计算过程可视化

import os
import shutil


image_infos = open("../config/test.txt").read().strip().split('\n')

if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_info in image_infos:
    image_boxes = image_info.split(' ')
    image = image_boxes[0]

    target_path = os.path.join("./input/images-optional", os.path.basename(image))
    shutil.copy(image, target_path)

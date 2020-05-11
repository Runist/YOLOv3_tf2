# -*- coding: utf-8 -*-
# @File : get_gt_txt.py
# @Author: Runist
# @Time : 2020/5/8 10:36
# @Software: PyCharm
# @Brief: 将真实图片转成mAP的txt检测格式
import os
import sys
import config.config as cfg

image_infos = open("../config/test.txt").read().strip().split('\n')


if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_info in image_infos:
    image_boxes = image_info.split(' ')
    image, boxes = image_boxes[0], image_boxes[1:]

    with open("./input/ground-truth/" + os.path.basename(image)[:-4] + ".txt", "w") as file:
        for box in boxes:
            xmin, ymin, xmax, ymax, idx = box.split(',')
            name = cfg.class_names[int(idx)]
            file.write("{} {} {} {} {}\n".format(name, xmin, ymin, xmax, ymax))

print("Conversion completed!")

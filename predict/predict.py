# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2020/4/16 10:38
# @Software: PyCharm
# @Brief: 模型预测
import config.config as cfg
import tensorflow as tf
import numpy as np
import colorsys
import os

from transform import parse_yolov3_output
from model.model import yolo_body

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from keras.models import load_model
from keras.layers import Input


class Yolov3Predict(object):

    def __init__(self, model_path):
        self.class_names = cfg.class_names
        self.anchors = cfg.anchors
        self.model_path = model_path

    def predict(self, image_path):
        """
        读取模型，做出预测，并处理预测结果。将其变成正常图片下的结果，而非416x416的结果
        :param image_path: 图片路径
        :return:
        """
        image, width, height = self.read_image(image_path)
        model = yolo_body()

        print("loading weights...")
        model.load_weights(self.model_path)

        output = model.predict(image)

        boxes, scores, classes = parse_yolov3_output(output, (height, width), max_boxes=20)
        return boxes, scores, classes

    def detect_image(self, image_path):
        """
        检测单张图片
        :param image_path: 图片路径
        """
        start = timer()

        # 读取预测结果
        out_boxes, out_scores, out_classes = self.predict(image_path)
        image = Image.open(image_path)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 框的厚度
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 获取坐标
            top, left, bottom, right = box
            top -= 5
            left -= 5
            bottom += 5
            right += 5

            # 防止小于0
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框、写上分类
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # 获取文字框的大小
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            # 如果文字框位置 小于 0，就在画面外边，这时候需要画在框上。在里面，就画在框上面
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 给定左上角 和 右下角坐标，画矩形
            draw.rectangle([left, top, right, bottom], outline=self.colors[c], width=thickness)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # 写上分类的文字
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print("use_time:{:.2f}s".format(end - start))

        return image

    @staticmethod
    def read_image(image_path):
        """
        读取图片，填充图片后归一化
        :param image_path: 图片路径
        :return: 图片的np数据、宽、高
        """
        image = Image.open(image_path)

        # 获取原图尺寸 和 网络输入尺寸
        image_w, image_h = image.size
        w, h = cfg.input_shape
        scale = min(w / image_w, h / image_h)
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)

        # 插值变换、填充图片
        image = image.resize((new_w, new_h), Image.BICUBIC)
        new_image = Image.new('RGB', cfg.input_shape, (128, 128, 128))
        new_image.paste(image, ((w - new_w) // 2, (h - new_h) // 2))

        # 归一化
        image_data = np.array(new_image, dtype=np.float32)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # 增加batch的维度

        return image_data, image.width, image.height


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    img_path = "D:/Python_Code/Tensorflow2.0/YOLOv3/VOCdevkit/VOC2012/JPEGImages/2008_007511.jpg"

    yolo = Yolov3Predict("../logs/yolo_test.ckpt")
    image = yolo.detect_image(img_path)
    image.show()

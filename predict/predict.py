# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2020/4/16 10:38
# @Software: PyCharm
# @Brief: 模型预测
import config.config as cfg
import numpy as np
import colorsys

from model.model import yolo_body
from PIL import Image, ImageFont, ImageDraw
from keras.models import load_model
from keras.layers import Input


class Yolov3Predict(object):
    def __init__(self, image_path, obj_threshold, nms_threshold):
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.path = image_path
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

    def _get_class(self):
        """
        get class name
        :return: class_names - list
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        get anchors (you can use anchors from config.py)
        :return: anchors - np.array
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        """
        生成模型，转换模型输出结果
        """
        model_path = os.path.expanduser(self.model_path)
        # assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body()
            self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        """
        检测单张图片
        :param image: 传入图片
        """
        start = timer()

        # 调整图片使其符合输入要求
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
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
    def letterbox_image(image, size):
        """
        填充图片，改变大小
        :param image: 图像
        :param size: 图片面积
        """
        image_w, image_h = image.size
        w, h = size
        scale = min(w / image_w, h / image_h)
        new_w = int(iw * scale)
        new_h = int(ih * scale)

        image = image.resize((new_w, new_h), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - new_w) // 2, (h - new_h) // 2))

        return new_image

    def read_image(self):
        img = Image.open(self.path)
        resize_image = self.letterbox_image(img, cfg.input_shape)
        image_data = np.array(resize_image, dtype=np.float32)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # 增加batch的维度

        return image_data

    if __name__ == '__main__':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        yolo = YOLO()

        path = "./VOCdevkit/VOC2012/JPEGImages/2011_000006.jpg"
        try:
            img = Image.open(path)
        except:
            print('Open Error! Try again!')
        else:
            result = yolo.detect_image(img)
            result.show()


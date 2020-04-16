# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/3/30 15:16
# @Software: PyCharm
# @Brief: YOLOv3数据读取 -- 用tf.data

import tensorflow as tf
import numpy as np
import config.config as cfg


class ReadYolo3Data:
    """
    tf.data.Dataset高速读取数据，提高GPU利用率
    """

    def __init__(self, data_path, input_shape, batch_size, max_boxes=20):
        self.data_path = data_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_boxes = max_boxes

    def read_data_and_split_data(self):
        with open(self.data_path, "r") as f:
            files = f.readlines()

        split = int(0.9 * len(files))
        train = files[:split]
        valid = files[split:]

        return train, valid

    def parse(self, annotation_line):
        """
        为tf.data.Dataset.map编写合适的解析函数，由于函数中某些操作不支持
        python类型的操作，所以需要用py_function转换，定义的格式如下
            Args:
              @param annotation_line: 是一行数据（图片路径 + 预测框位置）
        tf.py_function
            Args:
              第一个是要转换成tf格式的python函数，
              第二个输入的参数，
              第三个是输出的类型
        """
        image, bbox = tf.py_function(self.change_image_bbox, [annotation_line], [tf.float32, tf.int32])
        # py_function没有解析List的返回值，所以要拆包 再合起来传出去
        y_true_13, y_true_26, y_true_52 = tf.py_function(self.process_true_bbox, [bbox], [tf.float32, tf.float32, tf.float32])
        box_data = y_true_13, y_true_26, y_true_52

        # 如果py_function的输出有个[..., ...]那么结果也会是列表，一般单个使用的时候，可以不用加[]
        return image, box_data

    def change_image_bbox(self, annotation_line):
        """
        填充或缩小图片，因为有可能不是刚好416x416形状的，因为图片变了，预测框的坐标也要变
            Args:
              @param annotation_line: 一行数据
        """

        # line = annotation_line.split()
        line = str(annotation_line.numpy(), encoding="utf-8").split()
        image_path = line[0]
        bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        input_width, input_height = self.input_shape

        image_height_f = tf.cast(image_height, tf.float32)
        image_width_f = tf.cast(image_width, tf.float32)
        input_height_f = tf.cast(input_height, tf.float32)
        input_width_f = tf.cast(input_width, tf.float32)

        scale = min(input_width_f / image_width_f, input_height_f / image_height_f)
        new_height = image_height_f * scale
        new_width = image_width_f * scale

        # 将图片按照固定长宽比进行缩放 空缺部分 padding
        dx_f = (input_width - new_width) / 2
        dy_f = (input_height - new_height) / 2
        dx = tf.cast(dx_f, tf.int32)
        dy = tf.cast(dy_f, tf.int32)

        # 其实这一块不是双三次线性插值resize导致像素点放大255倍，原因是，无论是cv还是plt在面都i浮点数时，仅解释0-1完整比例
        image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
        new_image = tf.image.pad_to_bounding_box(image, dy, dx, input_height, input_width)

        # 生成image.shape的大小的全1矩阵
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, dy, dx, input_height, input_width)
        # 做个运算，白色区域变成0，填充0的区域变成1，再* 128，然后加上原图，就完成填充灰色的操作
        image = (1 - image_ones_padded) * 128 + new_image

        # 将图片归一化到0和1之间
        image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        # 为填充过后的图片，矫正bbox坐标
        box_data = np.zeros((self.max_boxes, 5))
        if len(bbox) > 0:
            # np.random.shuffle(bbox)
            if len(bbox) > self.max_boxes:
                bbox = bbox[:self.max_boxes]

            bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale + dx_f
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale + dy_f
            box_data[:len(bbox)] = bbox

        return image, box_data

    def process_true_bbox(self, box_data):
        # [anchors[mask] for mask in anchor_masks]

        # 维度(b, max_boxes, 5)还是一样的，只是换一下类型，换成float32
        true_boxes = np.array(box_data, dtype='float32')
        input_shape = np.array(self.input_shape, dtype='int32')  # 416,416

        # “...”(ellipsis)操作符，表示其他维度不变，只操作最前或最后1维。读出xy轴，读出长宽
        # true_boxes[..., 0:2] 是左上角的点 true_boxes[..., 2:4] 是右上角的点
        # 计算中心点 和 宽高
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # 实际的宽高 / 416 转成比例
        true_boxes[..., 0:2] = boxes_xy / input_shape
        true_boxes[..., 2:4] = boxes_wh / input_shape

        # 生成3种特征大小的网格
        grid_shapes = [input_shape // [32, 16, 8][i] for i in range(cfg.num_bbox)]
        # 创建3个特征大小的全零矩阵，[(b, 13, 13, 3, 25), ... , ...]存在列表中
        y_true = [np.zeros((grid_shapes[i][0], grid_shapes[i][1], cfg.num_bbox, 5 + cfg.num_classes),
                           dtype='float32') for i in range(cfg.num_bbox)]

        # 计算哪个先验框比较符合 真实框的Gw,Gh 以最高的iou作为衡量标准
        # 因为先验框数据没有坐标，只有宽高，那么现在假设所有的框的中心在（0，0），宽高除2即可。（真实框也要做一样的处理才能匹配）
        anchor_rightdown = cfg.anchors / 2.     # 网格中心为原点(即网格中心坐标为(0,0)),　计算出anchor 右下角坐标
        anchor_leftup = -anchor_rightdown     # 计算anchor 左上角坐标

        # 长宽要大于0才有效,也就是那些为了补齐到max_boxes大小的0数据无效
        # 返回一个列表，大于0的为True，小于等于0的为false
        # 选择具体一张图片，valid_mask存储的是true or false，然后只选择为true的行
        valid_mask = boxes_wh[..., 0] > 0
        # 只选择 > 0 的行
        wh = boxes_wh[valid_mask]
        wh = np.expand_dims(wh, 1)      # 在第二维度插入1个维度 (框的数量, 2) -> (框的数量, 1, 2)
        box_rightdown = wh / 2.
        box_leftup = -box_rightdown

        # 将每个真实框 与 9个先验框对比，刚刚对数据插入的维度可以理解为 每次取一个框出来shape（1,1,2）和anchors 比最大最小值
        # 所以其实可以看到源代码是有将anchors也增加一个维度，但在不给anchors增加维度也行。
        # 计算真实框和哪个先验框最契合，计算最大的交并比 作为最契合的先验框
        intersect_leftup = np.maximum(box_leftup, anchor_leftup)
        intersect_rightdown = np.minimum(box_rightdown, anchor_rightdown)
        intersect_wh = np.maximum(intersect_rightdown - intersect_leftup, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        # 计算真实框、先验框面积
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = cfg.anchors[..., 0] * cfg.anchors[..., 1]
        # 计算最大的iou
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchors = np.argmax(iou, axis=-1)

        # len(besh_anchor)是label中标了几个框，他就计算出几个，所以每个都要计算合适的先验框
        for key, value in enumerate(best_anchors):
            # 遍历三次（三种类型的框 对应 三个不同大小的特征层）
            for n in range(cfg.num_bbox):
                # 如果key（最优先验框的下表）
                if value in cfg.anchor_masks[n]:
                    # 真实框x * grid_shape的x，一般np.array都是（y,x）的格式
                    i = np.floor(true_boxes[key, 0] * grid_shapes[n][1]).astype('int32')
                    j = np.floor(true_boxes[key, 1] * grid_shapes[n][0]).astype('int32')

                    # 获取 先验框（二维列表）内索引
                    k = cfg.anchor_masks[n].index(value)
                    c = true_boxes[key, 4].astype('int32')

                    # 三个大小的特征层， 逐一赋值
                    y_true[n][j, i, k, 0:4] = true_boxes[key, 0:4]
                    y_true[n][j, i, k, 4] = 1    # 置信度是1 因为含有目标
                    y_true[n][j, i, k, 5+c] = 1  # 类别的one-hot编码，其他都为0

        return y_true

    def make_datasets(self, annotation, mode="train"):
        # 这是GPU读取方式
        # load train dataset
        dataset = tf.data.Dataset.from_tensor_slices(annotation)
        # map的作用就是根据定义的 函数，对整个数据集都进行这样的操作
        # 而不用自己写一个for循环，如：可以自己定义一个归一化操作，然后用.map方法都归一化
        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode == "train":
            # 打乱数据，这里的shuffle的值越接近整个数据集的大小，越贴近概率分布
            # 但是电脑往往没有这么大的内存，所以适量就好
            dataset = dataset.repeat().shuffle(buffer_size=10).batch(self.batch_size)
            # prefetch解耦了 数据产生的时间 和 数据消耗的时间
            # prefetch官方的说法是可以在gpu训练模型的同时提前预处理下一批数据
            dataset = dataset.prefetch(self.batch_size)
        else:
            dataset = dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)

        return dataset


if __name__ == '__main__':
    reader = ReadYolo3Data(cfg.annotation_path, cfg.input_shape, cfg.batch_size)
    train, valid = reader.read_data_and_split_data()
    train_datasets = reader.make_datasets(train)

    # reader.parse(train[0])

    image, bbox = next(iter(train_datasets))
    print(bbox[2].shape)

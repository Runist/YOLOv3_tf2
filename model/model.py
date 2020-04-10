# -*- coding: utf-8 -*-
# @File : model.py
# @Author: Runist
# @Time : 2020/3/30 13:44
# @Software: PyCharm
# @Brief: YOLO3模型实现

from model.darknet53 import darknet53, conv_bn_leaky
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, MaxPooling2D, Lambda, Input
import config.config as cfg


def conv_block_5_conv_block_2(inputs, filters):
    """
    5次（conv + bn + leaky激活）
    2次（conv + bn + leaky激活）
    :param inputs: 输入
    :param filters: 卷积核个数
    :return: x
    """
    x = conv_bn_leaky(inputs, filters, kernel_size=1)
    x = conv_bn_leaky(x, filters * 2, kernel_size=3)
    x = conv_bn_leaky(x, filters, kernel_size=1)
    x = conv_bn_leaky(x, filters * 2, kernel_size=3)
    output_5 = conv_bn_leaky(x, filters, kernel_size=1)

    x = conv_bn_leaky(output_5, filters * 2, kernel_size=3)
    output_7 = conv_bn_leaky(x, cfg.num_bbox * (cfg.num_classes+5), kernel_size=1, bn=False)

    # 这里不知道为什么要reshape成 (b, size, size, 3, num+5)才能让后面的yolo_loss中y_true生效
    # 应该是让y_true与y_pred默认一样
    # reshape_x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[1], num_bbox, num_classes + 5)))(x)

    return output_5, output_7


def conv_upsample(inputs, filters):
    """
    1次（conv + bn + leaky激活） + 上采样
    :param inputs: 输入层
    :param filters: 卷积核个数
    :return: x
    """
    x = conv_bn_leaky(inputs, filters, kernel_size=1)
    x = UpSampling2D(2)(x)

    return x


def yolo_body():
    """
    yolov3主体结构 用darknet53做特征提取，输出三个结果做目标框预测

    :return: model
    """
    height, width = cfg.input_shape
    input_image = Input(shape=(height, width, 3), dtype='float32')  # [b, 416, 416, 3]
    feat1, feat2, feat3 = darknet53(input_image)

    # 第三个先验框计算 5次卷积 + 2次卷积就可以输出结果
    conv_feat3, output3 = conv_block_5_conv_block_2(feat3, 512)

    # 从第三个分支的结果 -> 1x1卷积 -> 上采样 -> 和第二个分支的结果合并
    upsample_feat3 = conv_upsample(conv_feat3, 256)
    concat_feat2 = Concatenate()([upsample_feat3, feat2])

    # 第二个先验框计算 5次卷积 + 2次卷积就可以输出结果
    conv_feat2, output2 = conv_block_5_conv_block_2(concat_feat2, 256)

    # 从第二个分支的结果 -> 上采样 -> 和第一个分支的结果合并
    upsample_feat2 = conv_upsample(conv_feat2, 128)
    concat_feat1 = Concatenate()([upsample_feat2, feat1])

    # 第一个先验框计算，这边就不需要上采样了
    _, output1 = conv_block_5_conv_block_2(concat_feat1, 128)

    # 这里output1、output2、output3的shape分别是52x52, 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    # output3 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_3')(output3)
    # output2 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_2')(output2)
    # output1 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_1')(output1)

    model = Model(input_image, [output3, output2, output1])
    # model.summary()

    return model


def yolo_feat_reshape(feat):
    grid_size = tf.shape(feat)[1]
    reshape_feat = tf.reshape(feat, [-1, grid_size, grid_size, cfg.num_bbox, cfg.num_classes + 5])

    return reshape_feat


if __name__ == '__main__':
    inputs = tf.keras.layers.Input(shape=(416, 416, 3), dtype="float32")
    model_body = yolo_body()

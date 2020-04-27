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
    feat_52x52, feat_26x26, feat_13x13 = darknet53(input_image)

    # 13x13预测框计算 5次卷积 + 2次卷积就可以输出结果
    conv_feat_13x13, output_13x13 = conv_block_5_conv_block_2(feat_13x13, 512)

    # 13x13的特征层 -> 1x1卷积 -> 上采样 -> 和第26x26的特征层合并
    upsample_feat_26x26 = conv_upsample(conv_feat_13x13, 256)
    concat_feat26x26 = Concatenate()([upsample_feat_26x26, feat_26x26])

    # 26x26预测框计算 5次卷积 + 2次卷积就可以输出结果
    conv_feat_26x26, output_26x26 = conv_block_5_conv_block_2(concat_feat26x26, 256)

    # 26x26的特征层 -> 上采样 -> 和52x52的特征层合并
    upsample_feat_52x52 = conv_upsample(conv_feat_26x26, 128)
    concat_feat_52x52 = Concatenate()([upsample_feat_52x52, feat_52x52])

    # 52x52预测框计算，这边就不需要上采样了
    _, output_52x52 = conv_block_5_conv_block_2(concat_feat_52x52, 128)

    # 这里output1、output2、output3的shape分别是52x52, 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    output_52x52 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_3')(output_52x52)
    output_26x26 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_2')(output_26x26)
    output_13x13 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_1')(output_13x13)

    # 实际上13x13的感受野是比较大的，对应的是大的先验框
    # 相应的52x52感受野是比较小的，检测小物体，先验框也比较小
    model = Model(input_image, [output_13x13, output_26x26, output_52x52])
    model.summary()

    return model


def yolo_feat_reshape(feat):
    grid_size = tf.shape(feat)[1]
    reshape_feat = tf.reshape(feat, [-1, grid_size, grid_size, cfg.num_bbox, cfg.num_classes + 5])

    return reshape_feat


if __name__ == '__main__':
    inputs = tf.keras.layers.Input(shape=(416, 416, 3), dtype="float32")
    model_body = yolo_body()

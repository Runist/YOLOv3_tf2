# -*- coding: utf-8 -*-
# @File : train.py.py
# @Author: Runist
# @Time : 2020/4/7 12:37
# @Software: PyCharm
# @Brief:

import tensorflow as tf
import config.config as cfg
from dataset import ReadYolo3Data
from model.model import yolo_body
from loss import yolo_loss

from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3 import YoloV3


def build_model():
    yolo_model = yolo_body()
    # model = YoloV3(416, training=True, classes=cfg.num_classes)
    h, w = cfg.input_shape
    y_true = [Input(shape=(h // [32, 16, 8][i], w // [32, 16, 8][i], cfg.num_bbox, cfg.num_classes + 5)) for i in range(cfg.num_bbox)]

    loss_input = [y_true, yolo_model.output]
    model_loss = Lambda(yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss')(loss_input)

    # loss = [compute_loss(cfg.anchors[mask], num_classes=cfg.num_classes) for mask in cfg.anchor_masks]
    # model.compile(optimizer=Adam(lr=1e-3), loss=yolo_loss)

    model = Model(loss_input, model_loss)

    return model


def main():
    reader = ReadYolo3Data(cfg.annotation_path)
    train, valid = reader.read_data_and_split_data()
    train_datasets = reader.make_datasets(train)
    valid_datasets = reader.make_datasets(valid)

    image, y_true_13, y_true_26, y_true_52 = next(iter(train_datasets))
    bbox = [y_true_13, y_true_26, y_true_52]

    model = build_model()

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    # history = model.fit(train_datasets, epochs=10, validation_data=valid_datasets, callbacks=callbacks)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# @File : train.py.py
# @Author: Runist
# @Time : 2020/4/7 12:37
# @Software: PyCharm
# @Brief: 训练脚本

import tensorflow as tf
import config.config as cfg
from dataReader import ReadYolo3Data
from model.model import yolo_body
from loss import YoloLoss

import os
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, CategoricalAccuracy


def main():
    # 选择编号为0的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 读取数据
    reader = ReadYolo3Data(cfg.annotation_path, cfg.input_shape, cfg.batch_size)
    train, valid = reader.read_data_and_split_data()
    train_datasets = reader.make_datasets(train)
    valid_datasets = reader.make_datasets(valid)

    # 定义模型
    model = yolo_body()
    yolo_loss = [YoloLoss(cfg.anchors[mask]) for mask in cfg.anchor_masks]

    # 定义优化器、模型评估指标
    optimizer = Adam(learning_rate=cfg.learn_rating)
    train_loss = Mean(name='train_loss')
    valid_loss = Mean(name='valid_loss')

    # 设置保存最好模型的指标
    best_test_loss = float('inf')
    patience = 5
    min_delta = 1e-3
    patience_cnt = 0
    history_loss = []

    # low level 的方式计算loss
    for epoch in range(1, cfg.epochs + 1):
        step = 0
        print("Epoch {}/{}".format(epoch, cfg.epochs))
        total_step = len(train) // cfg.batch_size

        # 处理训练集数据
        for batch, (images, labels) in enumerate(train_datasets.take(total_step)):
            with tf.GradientTape() as tape:
                # 得到预测
                outputs = model(images, training=True)
                # 计算损失
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                # yolo_loss是怎么拆包到loss_fn？
                for output, label, loss_fn in zip(outputs, labels, yolo_loss):
                    pred_loss.append(loss_fn(label, output))
                total_train_loss = tf.reduce_sum(pred_loss) + regularization_loss

            # 反向传播梯度下降
            # model.trainable_variables代表把loss反向传播到每个可以训练的变量中
            grads = tape.gradient(total_train_loss, model.trainable_variables)
            # 将每个节点的误差梯度gradients，用于更新该节点的可训练变量值
            # zip是把梯度和可训练变量值打包成元组
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 更新train_loss
            train_loss.update_state(total_train_loss)
            # 输出训练过程
            rate = (step + 1) / total_step
            a = "*" * int(rate * 70)
            b = "." * int((1 - rate) * 70)
            loss = train_loss.result().numpy()
            print("\r{}/{} {:^3.0f}%[{}->{}] - loss:{:.4f}".
                  format(batch, total_step, int(rate * 70), a, b, loss), end='')
            step += 1
        print()

        # 计算验证集
        for batch, (images, labels) in enumerate(valid_datasets.take(total_step)):
            # 得到预测，不training
            outputs = model(images)
            # 正则化损失？
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, yolo_loss):
                pred_loss.append(loss_fn(label, output))
            total_valid_loss = tf.reduce_sum(pred_loss) + regularization_loss

            # 更新valid_loss
            valid_loss.update_state(total_valid_loss)
        print('Loss: {:.2f}, Test Loss: {:.2f}\n'.format(train_loss.result(), valid_loss.result()))
        # 保存loss，可以选择train的loss
        history_loss.append(valid_loss.result().numpy())

        # 只保存最好模型
        if valid_loss.result() < best_test_loss:
            best_test_loss = valid_loss.result()
            model.save_weights("./logs/{}-{}-{:.2f}.ckpt".
                               format(cfg.model_name, epoch, best_test_loss.numpy()), save_format='tf')

        # EarlyStopping
        if epoch > 1 and history_loss[epoch - 1] - history_loss[epoch] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            tf.print("No improvement for {} times, early stopping optimization".format(patience))
            break


if __name__ == '__main__':
    main()

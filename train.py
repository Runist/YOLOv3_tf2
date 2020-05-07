# -*- coding: utf-8 -*-
# @File : train.py.py
# @Author: Runist
# @Time : 2020/4/7 12:37
# @Software: PyCharm
# @Brief: 训练脚本

import tensorflow as tf
import shutil
import config.config as cfg
from core.dataReader import ReadYolo3Data
from core.loss import YoloLoss
from model.model import yolo_body

import os
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 读取数据
    reader = ReadYolo3Data(cfg.annotation_path, cfg.input_shape, cfg.batch_size)
    train, valid = reader.read_data_and_split_data()
    train_datasets = reader.make_datasets(train, "train")
    valid_datasets = reader.make_datasets(valid, "valid")
    train_steps = len(train) // cfg.batch_size
    valid_steps = len(valid) // cfg.batch_size

    yolo_loss = [YoloLoss(cfg.anchors[mask]) for mask in cfg.anchor_masks]

    # 清除summary目录下原有的东西
    for f in os.listdir(cfg.log_dir):
        file = os.path.join(cfg.log_dir, f)
        shutil.rmtree(file)

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train), len(valid), cfg.batch_size))
    if cfg.train_mode == "eager":
        # 定义优化器和学习率衰减速率
        # PolynomialDecay参数：cfg.learn_rating 经过 cfg.epochs 衰减到 cfg.learn_rating/10
        # 1、lr_fn是类似是一个函数，每次需要它来计算当前学习率都会调用它
        # 2、它具有一个内部计数器，每次调用apply_gradients，就会+1
        lr_fn = PolynomialDecay(cfg.learn_rating, cfg.epochs, cfg.learn_rating / 10, 2)
        optimizer = Adam(learning_rate=lr_fn)
        low_level_train(optimizer, yolo_loss, train_datasets, valid_datasets, train_steps, valid_steps)
    else:
        optimizer = Adam(learning_rate=cfg.learn_rating)
        high_level_train(optimizer, yolo_loss, train_datasets, valid_datasets, train_steps, valid_steps)


def create_model():
    """
    创建模型，方便MirroredStrategy的操作
    :return: Model
    """
    # 是否预训练
    if cfg.pretrain:
        print('Load weights {}.'.format(cfg.pretrain_weights_path))
        # 定义模型
        pretrain_model = tf.keras.models.load_model(cfg.pretrain_weights_path, compile=False)
        pretrain_model.trainable = False
        input_image = pretrain_model.input
        feat_52x52, feat_26x26, feat_13x13 = pretrain_model.layers[92].output, \
                                             pretrain_model.layers[152].output, \
                                             pretrain_model.layers[184].output
        model = yolo_body([input_image, feat_52x52, feat_26x26, feat_13x13])
    else:
        print("Train all layers.")
        model = yolo_body()

    return model


def low_level_train(optimizer, yolo_loss, train_datasets, valid_datasets, train_steps, valid_steps):
    """
    以底层的方式训练，这种方式更好地观察训练过程，监视变量的变化
    :param optimizer: 优化器
    :param yolo_loss: 自定义的loss function
    :param train_datasets: 以tf.data封装好的训练集数据
    :param valid_datasets: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param valid_steps: 同上
    :return: None
    """
    # 创建模型结构
    model = create_model()

    # 定义模型评估指标
    train_loss = Mean(name='train_loss')
    valid_loss = Mean(name='valid_loss')

    # 设置保存最好模型的指标
    best_test_loss = float('inf')
    patience = 10
    min_delta = 1e-3
    patience_cnt = 0
    history_loss = []

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=cfg.log_dir)

    # low level的方式计算loss
    for epoch in range(1, cfg.epochs + 1):
        train_loss.reset_states()
        valid_loss.reset_states()
        step = 0
        print("Epoch {}/{}".format(epoch, cfg.epochs))

        # 处理训练集数据
        for batch, (images, labels) in enumerate(train_datasets.take(train_steps)):
            with tf.GradientTape() as tape:
                # 得到预测
                outputs = model(images, training=True)
                # 计算损失(注意这里收集model.losses的前提是Conv2D的kernel_regularizer参数)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                # yolo_loss、label、output都是3个特征层的数据，通过for 拆包之后，一个loss_fn就是yolo_loss中一个特征层
                # 然后逐一计算,
                for output, label, loss_fn in zip(outputs, labels, yolo_loss):
                    pred_loss.append(loss_fn(label, output))

                # 总损失 = yolo损失 + 正则化损失
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
            rate = (step + 1) / train_steps
            a = "*" * int(rate * 70)
            b = "." * int((1 - rate) * 70)
            loss = train_loss.result().numpy()

            print("\r{}/{} {:^3.0f}%[{}->{}] - loss:{:.4f}".
                  format(batch, train_steps, int(rate * 100), a, b, loss), end='')
            step += 1

        # 计算验证集
        for batch, (images, labels) in enumerate(valid_datasets.take(valid_steps)):
            # 得到预测，不training
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, yolo_loss):
                pred_loss.append(loss_fn(label, output))

            total_valid_loss = tf.reduce_sum(pred_loss) + regularization_loss

            # 更新valid_loss
            valid_loss.update_state(total_valid_loss)

        print('\nLoss: {:.4f}, Test Loss: {:.4f}\n'.format(train_loss.result(), valid_loss.result()))
        # 保存loss，可以选择train的loss
        history_loss.append(valid_loss.result().numpy())

        # 保存到tensorboard里
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('valid_loss', valid_loss.result(), step=optimizer.iterations)

        # 只保存最好模型
        if valid_loss.result() < best_test_loss:
            best_test_loss = valid_loss.result()
            model.save_weights(cfg.model_path, save_format='tf')

        # EarlyStopping
        if epoch > 1 and history_loss[epoch - 2] - history_loss[epoch - 1] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            tf.print("No improvement for {} times, early stopping optimization.".format(patience))
            break


def high_level_train(optimizer, loss, train_datasets, valid_datasets, train_steps, valid_steps):
    """
    使用fit_generator方式训练，可以知道训练完的时间，以及更规范的添加callbacks参数
    :param optimizer: 优化器
    :param loss: 自定义的loss function
    :param train_datasets: 以tf.data封装好的训练集数据
    :param valid_datasets: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param valid_steps: 同上
    :return: None
    """
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=10, verbose=1),
        TensorBoard(log_dir=cfg.log_dir)
    ]
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model()
        model.compile(optimizer=optimizer, loss=loss)

    # initial_epoch用于恢复之前的训练
    model.fit(train_datasets,
              steps_per_epoch=max(1, train_steps),
              validation_data=valid_datasets,
              validation_steps=max(1, valid_steps),
              epochs=cfg.epochs,
              initial_epoch=0,
              callbacks=callbacks)

    model.save_weights(cfg.model_path)


if __name__ == '__main__':
    main()

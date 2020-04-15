# 使用Tensorflow2.0实现Yolov3 - 仍在完善中

### 进度

- [x] 使用tf.data读取数据
- [x] 模型搭建
- [x] 损失函数
- [x] 使用tf.GradientTape训练
- [ ] 使用model.fit方式训练
- [x] EarlyStopping, Save best model
- [x] GPU加速
- [ ] 迁移学习

### 一些问题和可能会出现的Bug

1. 不能保证loss在下降到平稳状态后，预测的准确率能达到比较高。
2. loss是参照论文的公式写的，与网上一些成熟的代码不一样，可能会出现梯度爆炸与梯度弥散。
3. 关于算法细节的地方，也没有很好的弄清楚，如果读者有发现bug，请及时提出。
4. 目前仅实现了传统的Yolov3，短期内还不会实现YoloTiny。

### Reference

- https://github.com/qqwweee/keras-yolo3
- https://github.com/bubbliiiing/yolo3-keras
	- 模型大体结构的构建
	- 损失函数

- https://github.com/zzh8829/yolov3-tf2
  - 模型构建的细节
  - tf.GradientTape训练的方式
  - 损失函数的结构

- https://github.com/aloyschen/tensorflow-yolo3
  - tf.data读取数据
  - 网络特征输出解码过程
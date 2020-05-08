# 使用Tensorflow2.0实现Yolov3 - 仍在完善中

### 进度

- [x] 使用tf.data读取数据
- [x] 模型搭建
- [x] 损失函数
- [x] 使用tf.GradientTape训练
- [x] 使用model.fit方式训练
- [x] Learning rate decay
- [x] tf.GradientTape下的callbacks
- [x] 迁移学习
- [x] 数据增强
- [x] kmeans提取先验框
- [x] Tensorboard
- [x] GPU加速
- [x] GPU分布式训练

### 一些问题和可能会出现的Bug

1. 不能保证loss在下降到平稳状态后，预测的准确率能达到比较高。
2. loss是参照论文的公式写的，与网上一些成熟的代码不一样，可能会出现梯度爆炸与梯度弥散。
3. 关于算法细节的地方，也没有很好的弄清楚，如果读者有发现bug，请及时提出。
4. 目前仅实现了传统的Yolov3，短期内还不会实现YoloTiny。

### 关于loss  function

​		由于个人习惯，我没有将Loss的定义书写在Yolo model中，而是单独放在core/loss.py中。其中各个部分的损失，我参阅了原论文以及网上的不同版本代码，在置信度和分类的损失上最终使用了sigmoid_cross_entropy_with_logits，在xy和wh的损失使用了sigmoid_cross_entropy_with_logits和均方差公式。其中qqwweee使用的是keras backend 子空间下的binary_crossentropy方法，但实际上最终的实现是sigmoid_cross_entropy_with_logits。所以在本系统中，二者无差异。

### 迁移学习与微调

​		因为之前的代码是采用tf1的版本，使用keras下Model的load_weights方法。这个方法是有一个skip_mismatch参数，他能自动地跳过一些不匹配的隐藏层。而在我仔细对照了用Cpp源码写的cfg配置文件后，发现确实与本系统中的网络结构不一样。而在tf2的版本下，load_weights并没有skip_mismatch方法，导致无法正确载入权重。而在本系统中，则只读取darknet53中的权重，只加载特征提取部分。具体效果如何，有待训练。

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

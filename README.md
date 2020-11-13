# Yolov3

## 已停止更新，请使用[更简洁的版本_Yolov3](https://github.com/Runist/YOLOv3)

### 原理

![yolov3.jpg](https://i.loli.net/2020/10/21/8g43mXl1ckaChMu.jpg)

YOLOv3由backbone(Darknet-53)和预测框回归部分组成。其中：

1. Darknet-53其重要特点是使用了**残差网络Residual**，darknet53中的残差卷积就是**进行一次3X3、步长为2的卷积，然后保存该卷积layer，再进行一次1X1的卷积和一次3X3的卷积，并把这个结果加上layer作为最后的结果**， 残差网络的特点是**容易优化**，并且能够通过增加相当的**深度来提高准确率**。其内部的**残差块使用了跳跃连接，缓解了在深度神经网络中增加深度带来的梯度消失问题。**图像输入Darknet-53后将会输出3个特征层，shape分别为(52, 52, 256)，(26, 26, 512)，(13, 13, 1024)。这三个特征层分别是检测小目标、中目标、大目标。
2. 在将模型输出结果解析成预测框坐标前，还需要进行特征融合。使得前面的特征层能获得更高级的特征。
3. 最后三个特征层输出的最后一个维度都为75，可以分解成3x(20+4+1)，3是一个特征层上的cell(特征层上的最小像素单位)指有3个先验框，20是因为VOC数据集中有20个分类，4代表预测框的中心坐标和宽高、1指的是框内物体的置信度

### 预测框解码

图片经过网络处理后得到预测框的中心坐标和宽高，但这个预测值是在特征层下的，我们需要将这个值进行解码。

![box regression.jpg](https://i.loli.net/2020/10/21/T4xzhIOU1roFCPR.jpg)

对于预测框输出一般使用四维向量$(t_x, t_y, t_w, t_h)$表示，分别表示窗口的中心点坐标和宽高。对于上图，红色的框A代表原始的先验框anchors，绿色的框G代表目标的真实框，我们的目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'，即：

- 给定anchor $A = (A_x, A_y, A_w, A_h)$和$GT=[G_x, G_y, G_w, G_h]$
- 寻找一种变换$F$，使得：$F(A_x,A_y,A_w,A_h)=(G'_x,G'_y,G'_w,G'_h)$，其中$(G'_x,G'_y,G'_w,G'_h)\approx(G_x,G_y,G_w,G_h)$

那么经过何种变换$F$才能从上图中的anchor $A$变为$G'$呢？ 比较简单的思路就是:

- 先做平移

$$
G'_x = sigmoid(t_x) + c_x\\
G'_y = sigmoid(t_y) + c_y
$$



- 再做尺度变换

$$
G'_w = A_w \cdot e^{t_w}\\
G'_h = A_h \cdot e^{t_h}
$$

当输入的anchor A与GT相差较小时，可以认为这种变换是一种线性变换， 那么就可以用线性回归来建模对窗口进行微调。

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
- [x] mAP的计算
- [x] 实时视频检测

### 一些问题和可能会出现的Bug

1. 在pycharm中，import不同文件夹下的包没有问题。但在linux的terminal中调用则会包ImportError。
	- 解决方法是import sys，之后添加sys.path.append("your project path")
2. 在做预测时需要将config.py中的pretrain置为False，否则模型无法正确加载。

### 关于loss  function

由于个人习惯，我没有将Loss的定义书写在Yolo model中，而是单独放在core/loss.py中。其中各个部分的损失，我参阅了原论文以及网上的不同版本代码，在置信度和分类的损失上最终使用了sigmoid_cross_entropy_with_logits，在xy和wh的损失使用了sigmoid_cross_entropy_with_logits和均方差公式。其中qqwweee使用的是keras backend 子空间下的binary_crossentropy方法，但实际上最终的实现是sigmoid_cross_entropy_with_logits。所以在本系统中，二者无差异。

### 迁移学习与微调

~~因为之前的代码是采用tf1的版本，使用keras下Model的load_weights方法。这个方法是有一个skip_mismatch参数，他能自动地跳过一些不匹配的隐藏层。而在我仔细对照了用Cpp源码写的cfg配置文件后，发现确实与本系统中的网络结构不一样。而在tf2的版本下，load_weights并没有skip_mismatch方法，导致无法正确载入权重。~~将tensorflow版本从2.0.0升级至2.2.0以上即可。在本系统中，则只读取darknet53中的权重，只加载特征提取部分。具体效果如何，有待训练。

### config - 配置文件

较为常用的配置文件一般是cfg、json格式的文件，因为没有原作者的框架复杂，所以在配置文件采用的是py格式，也方便各个文件的调用。

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

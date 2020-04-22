# -*- coding: utf-8 -*-
# @File : boxExtract.py
# @Author: Runist
# @Time : 2020/4/22 15:42
# @Software: PyCharm
# @Brief: 提取xml中的box信息

from xml.etree import ElementTree


class BoxExtract:
    def __init__(self, save_path, classes_path):
        """
        创建 图片路径 + box位置的 txt文本文件
        :param save_path: 输出txt保存位置
        :param classes_path: 分类信息保存位置
        """
        self.save_path = save_path
        self.label = None
        self.get_classes(classes_path)

    def get_classes(self, classes_path):
        """
        加载 分类信息
        :param classes_path: 文本路径
        :return: 分类数据
        """
        with open(classes_path) as f:
            class_names = f.readlines()
        self.label = [c.strip() for c in class_names]

    def build_map(self, image_path, label_path):
        """
        建立 图片与位置信息 的映射，信息将会保存在self.save_path中
        :return: None
        """
        # a为追加
        save_file = open(self.save_path, "a")

        for image, label in zip(os.listdir(image_path), os.listdir(label_path)):
            image_name = os.path.join(os.path.abspath(image_path), image)
            xml_name = os.path.join(label_path, label)

            boxes = self.extract_boxes(xml_name)

            # 把生成的二维列表，遍历一遍，拼成字符串
            labels = ""
            for b in boxes:
                labels += ','.join(b) + ' '

            map_information = "{} {}\n".format(image_name, labels)
            save_file.write(map_information)
        save_file.close()

    def extract_boxes(self, xml_path):
        """
        提取一个xml中的box个数和大小
        :param xml_path: xml的路径
        :return: boxes：存有所有box的四个点的信息
        """
        # 加载要解析的文件
        tree = ElementTree.parse(xml_path)
        # 获取文档的首部，可以理解为 数据结构中树的结构
        root = tree.getroot()
        # 提取每个边界框的信息
        boxes = list()

        # 然后用类似BeautifulSoup的findall()以Xpath语法查找，这是会返回一个列表，可以方便遍历
        for obj in root.findall('object'):
            name = obj.find('name').text

            # 不在识别类别里的不要，识别难度=1的也不要
            if name not in self.label:
                continue

            name = str(self.label.index(name))
            xmin = obj.find('bndbox/xmin').text
            ymin = obj.find('bndbox/ymin').text
            xmax = obj.find('bndbox/xmax').text
            ymax = obj.find('bndbox/ymax').text
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)

        return boxes


if __name__ == '__main__':
    box_extract = BoxExtract("../config/", "../config/voc_classes.txt")
    boxes = box_extract.extract_boxes("../../Tensorflow2.0/YOLOv3/VOCdevkit/VOC2012/Annotations/2007_000027.xml")
    print(boxes)
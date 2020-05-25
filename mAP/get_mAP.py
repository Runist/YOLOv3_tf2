# -*- coding: utf-8 -*-
# @File : get_mAP.py
# @Author: Runist
# @Time : 2020/5/9 9:20
# @Software: PyCharm
# @Brief: 计算模型mAP指标

import argparse
import shutil
import json
import glob
import math
import cv2
import os
import numpy as np
import operator


parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="可视化过程", action="store_true")
parser.add_argument('-np', '--no-plot', help="结果可视化", action="store_true")
parser.add_argument('-q', '--quiet', help="简化控制台输出.", action="store_true")
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="需要忽略的一些类别")
parser.add_argument('--set-class-iou', nargs='+', type=str, help="为某个特定的分类设定IOU阈值")
args = parser.parse_args()


def error(msg):
    """
    输出错误信息，退出程序
    :param msg: 错误信息
    :return:
    """
    print(msg)
    exit(0)


def file_lines_to_list(path):
    """
    将txt文件内容行转换为列表
    :param path:
    :return:
    """
    with open(path) as f:
        content = f.readlines()
    # 删除每行的空格 和 换行符\n
    content = [x.strip() for x in content]
    return content


def is_float_between_0_and_1(value):
    """
    确保参数处于是处于0-1的浮点数
    :param value:
    :return:
    """
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def draw_text_in_image(img, text, pos, color, line_width):
    """
    在图像中绘制文字
    :param img:
    :param text:
    :param pos:
    :param color:
    :param line_width:
    :return:
    """
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    """
    Plot - 调整轴
    :param r:
    :param t:
    :param fig:
    :param axes:
    :return:
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
    使用Matplotlib绘制图
    :param dictionary:
    :param n_classes: 分类个数
    :param window_title: 窗口名字
    :param plot_title: 图的标题
    :param x_label: x轴的标签
    :param output_path: 输出路径
    :param to_show: 是否显示
    :param plot_color: 图的颜色
    :param true_p_bar:
    :return:
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


MINOVERLAP = 0.5  # IOU默认值 (defined in the PASCAL VOC2012 challenge)
SHOW_TIMES = 1000  # 每张图片显示时间ms
args.no_animation = True
args.no_plot = False

# 如果没有要忽略的类，则用空列表替换
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# 确保cwd（）是python脚本的位置（以便每个路径都有意义）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')
# 图片的路径，如果没有，就不能可视化计算过程
IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')

if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            # 没找到图片
            args.no_animation = True
else:
    args.no_animation = True


# 如果用户未选择--no-animation选项，则尝试导入OpenCV
show_animation = False
if not args.no_animation:
    try:
        import cv2
        show_animation = True
    except ImportError:
        print("\"opencv-python\" not found, please install to visualize the results.")
        args.no_animation = True


# 如果用户未选择--no-plot选项，则尝试导入Matplotlib
draw_plot = False
if not args.no_plot:
    try:
        import matplotlib.pyplot as plt
        draw_plot = True
    except ImportError:
        print("\"matplotlib\" not found, please install it to get the resulting plots.")
        args.no_plot = True


# 创建一个 ".temp_files/" 和 "output/" 的目录结构
TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH):
    os.makedirs(TEMP_FILES_PATH)

output_files_path = "output"
if os.path.exists(output_files_path):
    # 清空输出目录
    shutil.rmtree(output_files_path)
os.makedirs(output_files_path)

if draw_plot:
    os.makedirs(os.path.join(output_files_path, "classes"))
if show_animation:
    os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

"""
 真实框处理
    每一个真实数据都加载到一个临时".json"文件中去
"""
# 获取真实框文本路径的列表
ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")

ground_truth_files_list.sort()
# 记录每个分类在计算mAP中出现的次数
gt_counter_per_class = {}
# 计算一张图片中出现的分类
counter_images_per_class = {}

gt_files = []
for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    # normpath规整路径，路径中最后的文件名
    file_id = os.path.basename(os.path.normpath(file_id))

    # 检查是否有对应的预测文件存在
    dr_path = os.path.join(DR_PATH, (file_id + ".txt"))
    if not os.path.exists(dr_path):
        error_msg = "Error. File not found: {}\n".format(dr_path)
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
        error(error_msg)

    # 转换每个gt的内容
    lines_list = file_lines_to_list(txt_file)

    # 创建存储真实框数据列表
    bounding_boxes = []
    already_seen_classes = []

    for line in lines_list:
        class_name, left, top, right, bottom = line.split()

        if class_name in args.ignore:
            continue

        bbox = left + " " + top + " " + right + " " + bottom
        bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

        if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
        else:
            # 如果不存在这个类，就置为1
            gt_counter_per_class[class_name] = 1

        if class_name not in already_seen_classes:
            if class_name in counter_images_per_class:
                counter_images_per_class[class_name] += 1
            else:
                counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)

    # 将 bounding_boxes 写入到生成的 ".json" 文件中
    new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
    gt_files.append(new_temp_file)
    with open(new_temp_file, 'w') as outfile:
        json.dump(bounding_boxes, outfile)


gt_classes = list(gt_counter_per_class.keys())
# 按照字母顺序排序
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)


"""
 Check format of the flag --set-class-iou (if used)
    e.g. check if class exists
"""
if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'

    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    # specific_iou_classes = ['class_1', 'class_2']
    specific_iou_classes = args.set_class_iou[::2]  # even
    # iou_list = ['IoU_1', 'IoU_2']
    iou_list = args.set_class_iou[1::2]  # 取出每个分类的iou，跳着取
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 预测框
    将每一个预测数据都加载到一个临时".json"文件中去
"""
# get a list with the detection-results files
dr_files_list = glob.glob(DR_PATH + '/*.txt')
dr_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in dr_files_list:
        # 首先还是检查对应的gt文件是否存在
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
        if class_index == 0:
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                error(error_msg)
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                dr_class_name, confidence, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error(error_msg)

            # 将每个图片的结果放进分类中，然后把分类写入json
            if dr_class_name == class_name:
                # print("match")
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox":bbox})
                # print(bounding_boxes)

    # 通过降低置信度对检测结果进行排序
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)


"""
 为每个分类计算AP
"""
sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
# open file to store the output
with open(output_files_path + "/output.txt", 'w') as output_file:
    output_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}

    # 遍历每个分类
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0

        # 从每个分类中加载预测结果

        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        # 将检测结果分配给真实物体
        nd = len(dr_data)
        tp = [0] * nd  # 创建一个大小为nd的零数组
        fp = [0] * nd

        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            # 如果要显示动画
            if show_animation:
                # 找到图片的路径
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                if len(ground_truth_img) == 0:
                    error("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    error("Error. Multiple image with id: " + file_id)
                else:

                    img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                    # 多次检测绘制图像
                    img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imread(img_cumulative_path)
                    else:
                        img_cumulative = img.copy()
                    # 为图片底部添加边框
                    bottom_border = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

            # 将检测结果分配给地面真实物体（如果有）
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # 加载检测的数据
            bb = [float(x) for x in detection["bbox"].split()]

            # 遍历每个真实框的数据
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # 计算IOU
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                             (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # 将检测结果分为 真阳性/ 假阴性 /假阳性
            if show_animation:
                status = "NO MATCH FOUND!"  # status只在可视化过程中中用
            # 设定最小iou阈值
            min_overlap = MINOVERLAP
            # 如果有对某个类单独设定的iou
            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])

            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # 真阳性
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                    # 更新到.json
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                    if show_animation:
                        status = "MATCH!"
                else:
                    # 假阳性（多次检测）
                    fp[idx] = 1
                    if show_animation:
                        status = "REPEATED MATCH!"
            else:
                # 假阳性
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

            """
             在图上画框，显示成动画
            """
            if show_animation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                    float(detection["confidence"]) * 100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0:  # 如果边界框之间有交集
                    bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                    cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                    cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                    cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # show image
                cv2.imshow("Animation", img)
                cv2.waitKey(SHOW_TIMES)  # 显示 20 ms
                # 保存图片输出
                output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" \
                                  + str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # 保存绘制有所有对象的图像
                cv2.imwrite(img_cumulative_path, img_cumulative)

        # 计算 准确的/召回率
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
        """
         写进 output.txt
        """
        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        if not args.quiet:
            print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr

        """
         Draw plot
        """
        if draw_plot:
            plt.plot(rec, prec, '-o')
            # add a new penultimate point to the list (mrec[-2], 0.0)
            # since the last line segment (and respective area) do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf() # gcf - get current figure
            fig.canvas.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            # plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca()  # 获取当前轴
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
            # 替代选项->等待按钮被按下
            # while not plt.waitforbuttonpress(): pass # wait for key display
            # 替代选项->正常显示
            # plt.show()
            # 保存图像
            fig.savefig(output_files_path + "/classes/" + class_name + ".png")
            plt.cla()  # 清除轴用于下一个绘图

    if show_animation:
        cv2.destroyAllWindows()

    output_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    output_file.write(text + "\n")
    print(text)


"""
 绘制假阳性的动画
"""
if show_animation:
    pink = (203, 192, 255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        # print(ground_truth_data)
        # 获取对应图像的名字
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
        img = cv2.imread(img_cumulative_path)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".jpg"
            img = cv2.imread(img_path)
        # 绘制假阳性的检测框
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), pink, 2)
        cv2.imwrite(img_cumulative_path, img)

# 删除临时生成的目录
shutil.rmtree(TEMP_FILES_PATH)

"""
 统计检测结果总数
"""

det_counter_per_class = {}
for txt_file in dr_files_list:
    lines_list = file_lines_to_list(txt_file)

    for line in lines_list:
        class_name = line.split()[0]
        # 检查类是否在忽略列表中，如果是，则跳过
        if class_name in args.ignore:
            continue
        # 计算目标出现的次数
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            det_counter_per_class[class_name] = 1

# print(det_counter_per_class)
dr_classes = list(det_counter_per_class.keys())


"""
 在真实框中绘制每个类别的发生总数
"""
if draw_plot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = output_files_path + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
        )


"""
 完成对真阳性的计数
"""
for class_name in dr_classes:
    # 如果类别存在于检测结果中，但不存在于真实框中，则该类别中没有真阳性
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
# print(count_true_positives)


"""
 在“detection-results”文件夹中绘制每个分类的总数
"""
if draw_plot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = output_files_path + "/detection-results-info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
        )


"""
 将每个类的检测到的对象数写入output.txt
"""
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        output_file.write(text)


"""
 绘制对数平均未命中率图（以降序显示所有类别的层数）
"""
if draw_plot:
    window_title = "lamr"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = output_files_path + "/lamr.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        lamr_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )


"""
 绘制mAP图（以降序显示所有类别的AP）
"""
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    x_label = "Average Precision"
    output_path = output_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )
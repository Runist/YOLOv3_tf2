# -*- coding: utf-8 -*-
# @File : video.py.py
# @Author: Runist
# @Time : 2020/5/11 14:09
# @Software: PyCharm
# @Brief: 调用摄像头进行实时检测

from tensorflow.keras.layers import Input
from predict.predict import Yolov3Predict
from PIL import Image
import numpy as np
import cv2 as cv
import config.config as cfg
import time

if __name__ == '__main__':
    yolo = Yolov3Predict(cfg.model_path)
    yolo.load_model()

    fps = 0.0
    cap = cv.VideoCapture(0)
    while True:
        t1 = time.time()

        ref, frame = cap.read()
        try:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % fps)
            frame = cv.putText(frame, "fps= %.2f" % fps, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
        cv.imshow("video", frame)
        c = cv.waitKey(30) & 0xff

        if c == 27:
            cap.release()
            break


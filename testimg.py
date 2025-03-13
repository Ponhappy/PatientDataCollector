from ultralytics import YOLO
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QComboBox,
    QLabel, QTextBrowser, QInputDialog, QMessageBox, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QVBoxLayout, QHBoxLayout
)
import sys
from PyQt5.QtWidgets import QWidget
from finger_detect import save_finger_pulse
from wrist_detect import save_wrist_pulse
import os
import threading
from wave import finger_PlotWidget
import serial
import serial.tools.list_ports
import json
from datetime import datetime
from wrist_thread import WristDataThread
from finger_thread import FingerDataThread
from camera_thread import CameraThread


def tongue_diagnosis(img):
    class_labels = {
        0: "您的舌质呈现粉红色，这通常与健康的舌象相符，表明您的身体状况良好，气血充足。粉红舌通常反映出良好的生理状态，然而，如果舌质偏红，则可能提示体内存在热症，需警惕潜在的炎症或感染情况。建议定期关注身体其他症状，保持健康的生活方式。",
        1: "您的舌苔薄白，通常表明体内没有明显的病理变化，可能处于健康状态。然而，薄白舌也可能提示气血不足或体内寒气较重，建议注意饮食调理，适当增加营养摄入，保持身体温暖，避免寒凉食物的过量摄入。",
        2: "您的舌苔厚白，这可能指示体内存在寒湿或痰湿，通常与消化系统功能障碍有关。厚白舌常见于脾胃虚弱、消化不良等情况，建议您关注饮食习惯，避免油腻和生冷食物，同时可以考虑适当的中药调理，以增强脾胃功能。",
        3: "您的舌苔厚黄，这通常表示体内有湿热，可能伴随发热、口渴、便秘等症状。厚黄舌常见于感染、炎症或消化系统疾病。建议您保持充足的水分摄入，避免辛辣刺激食物，同时可以考虑咨询专业医生进行进一步检查和调理。",
        4: "您的舌苔灰黑，这是一种较为严重的病理变化，可能与严重的感染、长期疾病、药物中毒或内脏器官的严重病变有关。灰黑舌通常提示体内存在较大的病理变化，建议您尽快就医，进行详细检查，以便及时发现并处理潜在的健康问题。"
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
    # model = YOLO(model_path)  # 加载模型
    # model_path = './runs/detect/train/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"模型文件不存在：{model_path}")
        return img, "模型文件不存在，请检查模型路径"

    model = YOLO(model_path)  # 加载模型
    results = model(img)
    if not results:
        print("未检测到舌像")
        return img, "未检测到舌像，请重新拍照"

    annotated_frame = results[0].plot()
    diagnosis = "没有发现舌像，请重新拍照"
    for result in results:
        class_ids = result.boxes.cls.numpy()  # 获取类别索引数组
        for class_id in class_ids:
            diagnosis = class_labels.get(int(class_id), "未知类别")

    print(diagnosis)
    return annotated_frame, diagnosis


img_path="./user_packages/30/snapshot_20250118-110036.jpg"
tongue_diagnosis()